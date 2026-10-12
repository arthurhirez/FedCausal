"""exact.py -- exact, threshold-free quantities of a probe stack, computed once and cached.

A probe stack holds K worlds of ONE city, world k converting district k. Every district's demand in a
world where it does not drift is the same series (the demand generator is keyed by seed, month and
node), so the city in which nothing drifts can be assembled exactly ("spliced") and simulated once.
Against it, every probe world's response is a PAIRED difference -- same weather, same month draws, same
hours -- with no estimation noise:

    dQ_k = Q^(k) - Q^(0)              the drift's effect on every link, hour by hour

On top of that this module computes, per link s and drifting district k (all on the stack's own
windows, so they are comparable to the stack's gains):

* exact labels      g_w (native gain on the weekly profile, oriented), S (shape effect), amp, Nr
* composition (E1)  where the link's water goes (complete mixing, flow-path destinations), in the
                    no-drift city (baseline window) and, paired, in probe world k vs the no-drift city
                    on k's settled window: dm_k, its two nulls and a paired weekly t
* storage share (E2) sigma = |g_ie0| / (|g_de0| + |g_ie0|), with g_de0 the response of the
                    mediator-blocked twin on the no-drift trajectory (demand transit alone) and
                    g_ie0 = g_tot - g_de0 the part that needs the hub (pumps, valves, tanks) to react;
                    undefined (NaN) where |g_de0| + |g_ie0| < ``sigma_floor``: a ratio of two near-zero
                    gains is not a channel split (it flipped 0.006 -> 0.63 between seeds on D-Town)
* topology          branch / source_bridge / loop / isolated, downstream sets of branches

and the gates that make the numbers trustworthy (G-X0 .. G-X6, ``gates.csv``).

Nothing here is a selection rule: no thresholds beyond numerical guards. Selection reads these
tables (``links.parquet``, ``labels.parquet``) and applies its own.

Cost: 1 + (K + 1 if e2) EPANET runs of the stack's horizon, plus E1 (~1-3 ms per network-hour).
Cache: ``<physics owner>/exact/<hash>/`` where the owner is the stack that holds the probe worlds
(``probes_from`` donor or the stack itself) and the hash covers the physics spec, this module's
config and ``EXACT_VERSION``. An analysis-only change of the stack (gate, mixture rule, thresholds)
therefore reuses the exact layer. ``save_series`` and ``sigma_floor`` are outside the hash: sigma is
recomputed from the stored g_de0 / g_ie0 on every load (:func:`sigma_share`), so a cached layer
always carries the floor it is loaded with.
"""
from __future__ import annotations

import json
import pickle
import time as _time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from fedwater.hashing import canonical_hash

from . import epanet_loop as el
from . import mixture_probe as mp
from . import signal_probe as sp
from .store import _read_json, _write_json, physics

__all__ = ["EXACT_VERSION", "DEFAULTS", "link_topology", "splice_nodrift", "E1Net",
           "sigma_share", "ensure_exact", "load_exact", "exact_dir"]

EXACT_VERSION = 1
DEFAULTS = {
    "e2": True,                 # storage share via blocked twins (K + 1 extra runs)
    "eps": 1e-9,                # L/s: a link below this carries no flow in E1
    "own_min": 1e-3,            # |g_w| on the home district below this: Nr undefined
    "sigma_floor": 1e-3,        # |g_de0| + |g_ie0| below this: sigma undefined (gain units, as own_min)
    "dead_frac": 1e-3,          # mean |q| below this x the median: a dead link
    "save_series": True,        # keep the no-drift city's flows (float32) for the dose notebook
    "gates": {"pre_drift": 1e-6, "branch_gain": 1e-3, "e1_row": 1e-9, "e1_close": 1e-6,
              "e1_branch": 1e-6, "twin": 1e-3, "branch_ie0": 1e-3},
}
GROUPS_TAIL = ("storage", "reservoir", "unaccounted")
_UNHASHED = ("save_series", "sigma_floor")      # applied on load / output only: never recompute for them


# ==========================================================================
# topology
# ==========================================================================
def link_topology(wn) -> tuple[pd.DataFrame, dict]:
    """Topological class of every link (as ``dependence_compute.topology``).

    A link that is a bridge of the network graph (not one of a parallel pair) splits it in two:
      branch         sources (tanks, reservoirs) on one side only: the other side is fed through
                     the link alone, so its flow is exactly that side's demand
      source_bridge  sources on both sides
      isolated       sources on neither side
    Every other link is ``loop``. ``down_sign`` is +1 when the downstream side holds the link's end
    node (the .inp orientation points downstream), else -1; ``down`` maps each branch to its
    downstream node set.
    """
    G = nx.Graph()
    pairs = Counter()
    ends = {}
    for ln in wn.link_name_list:
        lk = wn.get_link(ln)
        u, v = lk.start_node_name, lk.end_node_name
        ends[ln] = (u, v)
        G.add_edge(u, v)
        pairs[frozenset((u, v))] += 1
    sources = set(wn.tank_name_list) | set(wn.reservoir_name_list)
    bridges = {frozenset(e) for e in nx.bridges(G)}
    rows, down = [], {}
    for ln, (u, v) in ends.items():
        e = frozenset((u, v))
        base = {"element": ln, "start": u, "end": v, "link_type": type(wn.get_link(ln)).__name__}
        if e not in bridges or pairs[e] > 1:
            rows.append({**base, "topo": "loop", "down_sign": np.nan, "n_down": np.nan})
            continue
        G.remove_edge(u, v)
        cu, cv = nx.node_connected_component(G, u), nx.node_connected_component(G, v)
        G.add_edge(u, v)
        su, sv = bool(cu & sources), bool(cv & sources)
        if su and sv:
            rows.append({**base, "topo": "source_bridge", "down_sign": np.nan, "n_down": np.nan})
        elif su or sv:
            dn = cv if su else cu
            down[ln] = set(dn)
            rows.append({**base, "topo": "branch", "down_sign": 1.0 if v in dn else -1.0, "n_down": len(dn)})
        else:
            rows.append({**base, "topo": "isolated", "down_sign": np.nan, "n_down": np.nan})
    return pd.DataFrame(rows), down


# ==========================================================================
# the no-drift city
# ==========================================================================
def splice_nodrift(demands, districts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The no-drift city's demand from the probe worlds.

    ``demands``: iterable of (drifting district, demand frame). District d's columns are taken from
    the first world in which d does not drift; every other such world must carry the SAME columns
    (G-X1), and columns outside every district must agree across all worlds. Consumes the iterable
    once, holding one world's frame at a time.
    """
    node2d = {str(n): d for d, ns in districts.items() for n in ns}
    D0, have, diff = None, set(), Counter()
    ndon = Counter()
    for k, D in demands:
        D = _cols(D)
        if D0 is None:
            D0 = D.copy()
            cols = [c for c in D0.columns if c != "month"]
            kc = {d: [c for c in cols if node2d.get(c) == d] for d in districts}
            other = [c for c in cols if c not in node2d]
        for d in districts:
            if d == k:
                continue
            ndon[d] += 1
            if d not in have:
                D0[kc[d]] = D[kc[d]].to_numpy()
                have.add(d)
            else:
                diff[d] = max(diff[d], float(np.abs(D[kc[d]].to_numpy(float)
                                                    - D0[kc[d]].to_numpy(float)).max(initial=0.0)))
        if other:
            ndon["(no district)"] += 1
            diff["(no district)"] = max(diff["(no district)"], float(np.abs(
                D[other].to_numpy(float) - D0[other].to_numpy(float)).max(initial=0.0)))
    missing = [d for d in districts if d not in have]
    if missing:
        raise ValueError(f"splice: no world in which {missing} stay(s) put (need K >= 2 worlds)")
    rows = [{"district": d, "columns": len(kc.get(d, other)), "donors": ndon[d], "max_abs_diff": diff[d]}
            for d in [*districts, *(["(no district)"] if other else [])]]
    return D0, pd.DataFrame(rows)


def _cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c if c == "month" else str(c) for c in df.columns]
    return df.reset_index(drop=True)


# ==========================================================================
# E1 -- flow-path destinations (complete mixing), validated in epanet_poc / epanet_gt / recovery
# ==========================================================================
class E1Net:
    """Destination decomposition of every link's water, hour by hour.

    With phi_n the destination shares of the water leaving node n and O_n its total outflow (links +
    demand + tank sink):   O_n phi_n = d_n e_district(n) + s_n e_storage + sum_{n->v} q phi_v.
    A link carries the shares of its downstream node. Destinations are grouped as
    [district_1 .. district_K, storage (filling tanks), reservoir (water flowing back into one),
    unaccounted (inflow a junction does not pass on: the solver's imbalance, measured)].
    """

    def __init__(self, wn, links: list[str], demand_cols: list[str], districts: dict, eps: float = 1e-9):
        self.names = list(districts)
        node2d = {str(n): d for d, ns in districts.items() for n in ns}
        ends = {ln: (wn.get_link(ln).start_node_name, wn.get_link(ln).end_node_name) for ln in links}
        nodes = sorted({x for e in ends.values() for x in e})
        ix = {n: i for i, n in enumerate(nodes)}
        K = len(self.names)
        self.G = K + 3
        self.links, self.eps = list(links), float(eps)
        self.st = np.array([ix[ends[l][0]] for l in links])
        self.en = np.array([ix[ends[l][1]] for l in links])
        self.jcols = [c for c in demand_cols if c in ix]
        self.jidx = np.array([ix[c] for c in self.jcols], dtype=int)
        di = {d: i for i, d in enumerate(self.names)}
        self.jdst = np.array([di.get(node2d.get(c), -1) for c in self.jcols], dtype=int)
        self.tix = [ix[t] for t in wn.tank_name_list if t in ix]
        self.rix = [ix[r] for r in wn.reservoir_name_list if r in ix]
        special = set(self.tix) | set(self.rix)
        self.junc_like = np.array([i not in special for i in range(len(nodes))])
        self.N = len(nodes)

    def step(self, f: np.ndarray, d: np.ndarray):
        """One hour. f: flows (L/s, .inp sign) in ``links`` order; d: demand in ``jcols`` order.
        Returns (active link idx, contributions n_act x G, |q| of active, row-sum deviation)."""
        from scipy.sparse import csr_matrix, diags
        from scipy.sparse.linalg import splu
        K = len(self.names)
        act = np.abs(f) > self.eps
        fwd = f > 0
        u = np.where(fwd, self.st, self.en)[act]
        v = np.where(fwd, self.en, self.st)[act]
        q = np.abs(f)[act]
        N = self.N
        O = np.zeros(N)
        np.add.at(O, u, q)
        qin = np.zeros(N)
        np.add.at(qin, v, q)
        S = np.zeros((N, self.G))
        dn = np.zeros(N)
        dn[self.jidx] = d
        ok = self.jdst >= 0
        S[self.jidx[ok], self.jdst[ok]] += d[ok]
        O += dn
        for col, idxs in ((K, self.tix), (K + 1, self.rix)):     # a filling tank / receiving reservoir
            for ti in idxs:
                sink = max(qin[ti] - (O[ti] - dn[ti]), 0.0)
                S[ti, col] += sink
                O[ti] += sink
        excess = np.where(self.junc_like, np.clip(qin - O, 0.0, None), 0.0)
        S[:, K + 2] += excess
        O += excess
        dead = O <= self.eps
        A = diags(np.where(dead, 1.0, O)) - csr_matrix((q, (u, v)), shape=(N, N))
        Phi = splu(A.tocsc()).solve(S)
        rs = Phi.sum(axis=1)[~dead]
        dev = float(np.max(np.abs(rs - 1.0))) if len(rs) else 0.0
        return np.where(act)[0], q[:, None] * Phi[v], q, dev

    def run(self, F: np.ndarray, D: np.ndarray, hours, sd: int, how_window=None):
        """Accumulate over ``hours`` (absolute indices into F / D rows).

        Returns dict: day (n_days x L x G float32, keyed by absolute day in ``days``), days, how
        (L x 7sd x G hour-of-week totals over ``how_window`` hours, or None), row_dev, close_dev."""
        hours = np.asarray(sorted(hours))
        days = np.unique(hours // sd)
        dpos = {dd: i for i, dd in enumerate(days)}
        L = len(self.links)
        day = np.zeros((len(days), L, self.G), np.float32)
        acc = np.zeros((L, self.G))
        cur = None
        P = 7 * sd
        how = np.zeros((L, P, self.G)) if how_window is not None else None
        hw = how_window or (0, 0)
        row_dev, close_dev = 0.0, 0.0
        for t in hours:
            dd = t // sd
            if cur is not None and dd != cur:
                day[dpos[cur]] = acc
                acc[:] = 0.0
            cur = dd
            li, c, q, dev = self.step(F[t], D[t])
            row_dev = max(row_dev, dev)
            if len(q):
                close_dev = max(close_dev, float(np.abs(c.sum(axis=1) - q).max() / max(q.max(), 1e-12)))
            acc[li] += c
            if how is not None and hw[0] <= t < hw[1]:
                how[li, t % P] += c
        if cur is not None:
            day[dpos[cur]] = acc
        return {"day": day, "days": days, "how": how, "row_dev": row_dev, "close_dev": close_dev}


def _window(day: np.ndarray, days: np.ndarray, lo: int, hi: int, sd: int) -> np.ndarray:
    sel = (days >= lo // sd) & (days < hi // sd)
    return day[sel].astype(float).sum(axis=0)


def _weekly(day: np.ndarray, days: np.ndarray, lo: int, hi: int, sd: int) -> np.ndarray:
    """(n_weeks, L, G) totals of whole 7-day blocks from the window start (a partial tail dropped)."""
    sel = (days >= lo // sd) & (days < hi // sd)
    x = day[sel].astype(float)
    nw = len(x) // 7
    return x[:nw * 7].reshape(nw, 7, *x.shape[1:]).sum(axis=1)


def _renorm(C: np.ndarray, K: int) -> np.ndarray:
    """District shares among the district destinations (last axis), NaN where none."""
    s = C[..., :K].sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, C[..., :K] / np.where(s > 0, s, 1.0), np.nan)


def _raw(C: np.ndarray) -> np.ndarray:
    s = C.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, C / np.where(s > 0, s, 1.0), np.nan)


# ==========================================================================
# paired effect on one window
# ==========================================================================
def _profile(X: np.ndarray, sd: int, lo: int) -> np.ndarray:
    X = np.asarray(X, float)
    return np.atleast_2d(sp.weekly_profile(X if X.ndim > 1 else X[:, None], sd, step0=lo))


def _zrows(A: np.ndarray) -> np.ndarray:
    s = A.std(axis=1, keepdims=True)
    return (A - A.mean(axis=1, keepdims=True)) / np.where(s > 1e-12, s, np.nan)


def _gain_w(dQ: np.ndarray, dD: np.ndarray, sd: int, lo: int) -> np.ndarray:
    """<P dQ, P dD> / ||P dD||^2 per column (unoriented), P = weekly profile."""
    PQ, PD = _profile(dQ, sd, lo), _profile(dD, sd, lo)[0]
    n = float(PD @ PD)
    return PQ @ PD / n if n > 1e-18 else np.full(PQ.shape[0], np.nan)


def paired_effect(Q1, Q0, dD, sd: int, lo: int, orient) -> dict:
    """Exact effect of k's drift on every link over one window (rows = hours of the window).

    g_w   native gain on the weekly profile, oriented by the no-drift mean sign: extra L/s through the
          link per extra L/s of k's demand, in the part of the response that follows k's weekly pattern
    S     shape effect: rms over the week of z(P Q1) - z(P Q0), in sd of the weekly profile (what a client
          sees after per-client scaling); 0 where nothing changed, NaN where a profile is flat
    amp   ||dQ|| / ||dD|| over the window's hours
    """
    Q1, Q0 = np.asarray(Q1, float), np.asarray(Q0, float)
    dQ, dD = Q1 - Q0, np.asarray(dD, float)
    mdq = np.abs(dQ).mean(axis=0)
    Z = _zrows(_profile(Q1, sd, lo)) - _zrows(_profile(Q0, sd, lo))
    with np.errstate(invalid="ignore"):
        S = np.sqrt(np.nanmean(Z ** 2, axis=1))
    S = np.where(np.isnan(S) & (mdq == 0), 0.0, S)
    nD = float(np.sqrt(dD @ dD))
    return {"g_w": _gain_w(dQ, dD, sd, lo) * orient, "S": S,
            "amp": np.sqrt((dQ ** 2).sum(axis=0)) / max(nD, 1e-12), "mean_abs_dq": mdq}


def _orient(Q: np.ndarray) -> np.ndarray:
    s = np.sign(np.asarray(Q, float).mean(axis=0))
    return np.where(s == 0, 1.0, s)


def _rel_dev(A, B) -> np.ndarray:
    """Per column: max |A - B| / max |B|."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return np.abs(A - B).max(axis=0) / np.maximum(np.abs(B).max(axis=0), 1e-9)


# ==========================================================================
# stack access
# ==========================================================================
def _physics_owner(stack_path: Path) -> Path:
    mf = _read_json(stack_path / "manifest.json") or {}
    return stack_path.parent / mf["probes_from"] if mf.get("probes_from") else stack_path


def _world(src: Path, d: str, doc: dict, with_wn: bool = False, flows: bool = True) -> sp.Probe:
    """A probe world without pressures (the exact layer never reads them)."""
    wdir = src / "worlds" / d
    if not (wdir / "params.yml").exists():
        raise FileNotFoundError(f"{wdir}: probe world not retained (probe.retain_series was false)")
    wn = None
    if with_wn:
        with open(wdir / "wn.pkl", "rb") as fh:
            wn = pickle.load(fh)
    return sp.Probe.from_world({
        "params": yaml.safe_load((wdir / "params.yml").read_text()), "districts": doc,
        "demand": _cols(pd.read_parquet(wdir / "demand.parquet")), "pressures": pd.DataFrame(),
        "flows": _cols(pd.read_parquet(wdir / "flows.parquet")) if flows else pd.DataFrame(),
        "schedule": pd.read_csv(wdir / "schedule.csv", dtype={"node": str}),
        "validation": None, "wn": wn, "label": f"{src.name}:{d}"})


def stack_windows(P: sp.Probe, settle: dict, season_aligned: bool) -> tuple[tuple, tuple]:
    """(baseline, settled) step windows exactly as ``mixture_probe.build_mixture`` measures world P."""
    if season_aligned:
        return mp.season_windows(P, pad=int(settle["pad"]))
    return mp.baseline_window(P), mp.settled_window(P, ref_months=int(settle["ref_months"]),
                                                   slack=float(settle["slack"]), pad=int(settle["pad"]))


def exact_dir(stack_path, cfg: dict | None = None) -> Path:
    stack_path = Path(stack_path)
    cfg = {**DEFAULTS, **(cfg or {})}
    spec = (_read_json(stack_path / "manifest.json") or {}).get("spec", {})
    h = canonical_hash({"physics": physics(spec), "exact": {k: v for k, v in cfg.items() if k not in _UNHASHED},
                        "version": EXACT_VERSION})
    return _physics_owner(stack_path) / "exact" / h


def sigma_share(g_de0, g_ie0, floor: float) -> np.ndarray:
    """Storage share |g_ie0| / (|g_de0| + |g_ie0|); NaN where the two gains sum below ``floor``."""
    a, b = np.abs(np.asarray(g_de0, float)), np.abs(np.asarray(g_ie0, float))
    den = a + b
    ok = den >= max(float(floor), 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, b / np.where(ok, den, 1.0), np.nan)


def load_exact(path, cfg: dict | None = None) -> dict:
    """A cached exact layer. ``cfg`` (the ``sensor_placement.exact`` block) supplies ``sigma_floor``;
    sigma is recomputed from the stored gains with it (DEFAULTS when absent)."""
    path = Path(path)
    mf = _read_json(path / "manifest.json") or {}
    if mf.get("status") != "ok":
        raise ValueError(f"{path}: exact layer status {mf.get('status')!r}: {mf.get('error', '')}")
    out = {"path": path, "manifest": mf}
    for name in ("links", "labels", "downstream"):
        out[name] = pd.read_parquet(path / f"{name}.parquet")
    for name in ("gates", "runs", "splice"):
        out[name] = pd.read_csv(path / f"{name}.csv")
    floor = float({**DEFAULTS, **(cfg or {})}["sigma_floor"])
    L = out["labels"]
    if {"g_de0", "g_ie0"} <= set(L.columns):
        L["sigma"] = sigma_share(L["g_de0"], L["g_ie0"], floor)
    out["manifest"] = {**mf, "sigma_floor": floor}
    return out


# ==========================================================================
# the layer
# ==========================================================================
def ensure_exact(stack_path, cfg: dict | None = None, force: bool = False, verbose: bool = True) -> dict:
    """Load the exact layer of a probe stack, or compute and cache it. See the module docstring."""
    stack_path = Path(stack_path)
    cfg = {**DEFAULTS, **(cfg or {})}
    gth = {**DEFAULTS["gates"], **(cfg.get("gates") or {})}
    out_dir = exact_dir(stack_path, cfg)
    mf = _read_json(out_dir / "manifest.json")
    if mf and mf.get("status") == "ok" and not force:
        if verbose:
            print(f"exact layer {out_dir.relative_to(out_dir.parents[2])}: cached")
        return load_exact(out_dir, cfg)

    t_start = _time.time()
    smf = _read_json(stack_path / "manifest.json") or {}
    if smf.get("status") != "ok":
        raise ValueError(f"{stack_path}: stack status {smf.get('status')!r}")
    spec = smf["spec"]
    settle = spec["probe"]["settle"]
    season_aligned = bool(spec.get("plan", {}).get("season_aligned", False))
    src = _physics_owner(stack_path)
    dy = yaml.safe_load((stack_path / "districts.yml").read_text())
    districts = {d: [str(n) for n in ns] for d, ns in dy["districts"].items()}
    doc = {"districts": districts, "assets": dy.get("assets") or {}}
    names = list(districts)
    K = len(names)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"version": EXACT_VERSION, "stack": stack_path.name, "physics_owner": src.name,
                "network": smf["network"], "method": smf["method"], "districts": names,
                "cfg": cfg, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        res = _compute(src, doc, districts, names, K, settle, season_aligned, cfg, gth, out_dir, verbose)
        manifest.update(res["manifest"])
        for name in ("links", "labels", "downstream"):
            res[name].to_parquet(out_dir / f"{name}.parquet", index=False)
        for name in ("gates", "runs", "splice"):
            res[name].to_csv(out_dir / f"{name}.csv", index=False)
        manifest.update({"status": "ok", "error": "", "seconds": round(_time.time() - t_start, 1),
                         "gates_passed": bool(res["gates"]["passed"].all())})
        _write_json(out_dir / "manifest.json", manifest)
    except Exception as exc:
        _write_json(out_dir / "manifest.json", {**manifest, "status": "failed",
                                                "error": f"{type(exc).__name__}: {exc}"[:2000],
                                                "seconds": round(_time.time() - t_start, 1)})
        raise
    if verbose:
        g = res["gates"]
        print(f"exact layer {smf['network']}/{smf['method']}/{stack_path.name}: "
              f"{manifest['seconds'] / 60:.1f} min | gates {int(g['passed'].sum())}/{len(g)} passed")
    return load_exact(out_dir, cfg)


def _compute(src, doc, districts, names, K, settle, season_aligned, cfg, gth, out_dir, verbose) -> dict:
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t0 = _time.time()
    runs = []

    def log(r, run):
        runs.append({"run": run, "completed": bool(r["completed"]), "t_end_h": r["t_end_h"],
                     "n_steps": r["n_steps"], "unbalanced_steps": int(r["warnings"].get(1, 0)),
                     "warnings": json.dumps({str(k): v for k, v in r["warnings"].items()}),
                     "seconds": round(r["seconds"], 1)})
        return r

    # -- pass 1: splice the no-drift city; windows and model identity --------------------------------
    win, models = {}, {}

    def demands():
        for d in names:
            P = _world(src, d, doc, with_wn=True, flows=(d == names[0]))
            win[d] = stack_windows(P, settle, season_aligned)
            models[d] = canonical_hash(_model_signature(P.wn))
            if d == names[0]:
                win["_P0"] = P
            yield d, P.demand
    D0, splice = splice_nodrift(demands(), districts)
    P0 = win.pop("_P0")
    wn0 = P0.wn
    sd, sm = P0.steps_day, P0.steps_month
    H = len(D0)
    jcols = [c for c in D0.columns if c != "month"]
    kc = {d: [c for c in jcols if c in set(districts[d])] for d in names}
    Dk0 = {d: D0[kc[d]].to_numpy(float).sum(axis=1) for d in names}
    say(f"  splice: {len(jcols)} demand columns, max donor diff {splice['max_abs_diff'].max():.1e} "
        f"({_time.time() - t0:.0f} s)")

    # -- static link table -------------------------------------------------------------------------
    TOPO, DOWN = link_topology(wn0)
    links = [l for l in wn0.link_name_list]
    lix = {l: i for i, l in enumerate(links)}
    pumps, valves = set(wn0.pump_name_list), set(wn0.valve_name_list)
    cand = sp.candidates(P0, kinds=("flow",)).set_index("element")
    # the no-drift city never drifts: its baseline is the widest of the probes' baselines
    W0 = max((win[d][0] for d in names), key=lambda w: w[1] - w[0])

    # -- no-drift run ------------------------------------------------------------------------------
    r0 = log(el.simulate(wn0, D0, H, pmin_nodes=jcols, dtype=np.float32), "no-drift")
    if r0["links"] != links:
        raise ValueError("link order of the toolkit run differs from the model's")
    R0 = r0["snap"]
    say(f"  no-drift city simulated ({r0['seconds']:.0f} s, completed {r0['completed']})")
    if cfg.get("save_series", True):
        pd.DataFrame(R0, columns=links).to_parquet(out_dir / "nodrift_flows.parquet", index=False)

    # -- E1 on the no-drift city: baseline window + every settled window (paired) ---------------------
    e1 = E1Net(wn0, links, jcols, districts, eps=float(cfg["eps"]))
    Dv0 = D0[e1.jcols].to_numpy(float)
    hrs = set(range(*W0))
    for d in names:
        hrs |= set(range(*win[d][1]))
    E0 = e1.run(R0.astype(float), Dv0, hrs, sd, how_window=W0)
    say(f"  E1 no-drift: {len(hrs)} h ({_time.time() - t0:.0f} s)")
    e1_gate = [{"world": "no-drift", "row_dev": E0["row_dev"], "close_dev": E0["close_dev"]}]

    # baseline composition per link
    C0 = _window(E0["day"], E0["days"], *W0, sd)                     # L x G
    vol0 = C0.sum(axis=1)
    c0, m0 = _raw(C0), _renorm(C0, K)
    mean_abs = np.abs(R0[W0[0]:W0[1]].astype(float)).mean(axis=0)
    pos = mean_abs[mean_abs > 0]
    dead = mean_abs < float(cfg["dead_frac"]) * (np.median(pos) if len(pos) else 1.0)
    how = E0["how"]                                                   # L x P x G
    m_how, c_how = _renorm(how, K), _raw(how)

    LT = TOPO.set_index("element").reindex(links)
    rows = []
    for i, l in enumerate(links):
        home = cand["district"].get(l) if l in cand.index else None
        hi = names.index(home) if home in names else None
        r = {"element": l, "start": LT.at[l, "start"], "end": LT.at[l, "end"], "link_type": LT.at[l, "link_type"],
             "is_pump": l in pumps, "is_valve": l in valves, "topo": LT.at[l, "topo"],
             "down_sign": LT.at[l, "down_sign"], "n_down": LT.at[l, "n_down"],
             "home": home, "role": cand["role"].get(l) if l in cand.index else None,
             "mean_abs_flow": mean_abs[i], "dead": bool(dead[i]), "volume": vol0[i]}
        for j, d in enumerate(names):
            r[f"c_{d}"], r[f"m_{d}"] = c0[i, j], m0[i, j]
        for j, g in enumerate(GROUPS_TAIL):
            r[f"c_{g}"] = c0[i, K + j]
        mi = m0[i]
        if np.isfinite(mi).all():
            p = np.clip(mi, 1e-12, 1.0)
            r["n_eff"] = float(np.exp(-(p * np.log(p)).sum()))
        else:
            r["n_eff"] = np.nan
        r["m_home"] = mi[hi] if hi is not None else np.nan
        if hi is not None and not dead[i]:
            hs = m_how[i, :, hi]
            hr = c_how[i, :, hi]
            ok = np.isfinite(hs)
            r["swing_renorm"] = float(np.nanquantile(hs, 0.95) - np.nanquantile(hs, 0.05)) if ok.any() else np.nan
            r["swing_raw"] = float(np.nanquantile(hr, 0.95) - np.nanquantile(hr, 0.05)) if ok.any() else np.nan
            if ok.any() and np.isfinite(mi).all():
                r["flip_share"] = float((np.nanargmax(np.nan_to_num(m_how[i, ok], nan=-1), axis=1)
                                         != int(np.argmax(mi))).mean())
            else:
                r["flip_share"] = np.nan
        else:
            r["swing_renorm"] = r["swing_raw"] = r["flip_share"] = np.nan
        if l in DOWN:
            dn = DOWN[l]
            cols = [c for c in jcols if c in dn]
            r["down_demand"] = float(D0[cols].iloc[W0[0]:W0[1]].to_numpy(float).sum(axis=1).mean()) if cols else 0.0
        else:
            r["down_demand"] = np.nan
        rows.append(r)
    LINKS = pd.DataFrame(rows)

    # G-X5: E1 branch composition = downstream demand shares (exact)
    br = [l for l in links if l in DOWN]
    bdev = 0.0
    for l in br:
        i = lix[l]
        if dead[i] or not np.isfinite(m0[i]).all():
            continue
        ex = np.array([D0[[c for c in kc[d] if c in DOWN[l]]].iloc[W0[0]:W0[1]].to_numpy(float).sum()
                       for d in names])
        if ex.sum() > 0:
            bdev = max(bdev, float(np.abs(m0[i] - ex / ex.sum()).max()))

    # -- E2: the twin on the no-drift trajectory --------------------------------------------------
    e2 = bool(cfg["e2"])
    gaps = 0
    if e2:
        wb, dem, info = el.blocked_model(wn0, D0, r0, H)
        gaps = int(info["ill_posed_hours"])
        b0 = log(el.simulate(wb, dem, H, pmin_nodes=jcols, dtype=np.float32), "twin(0, traj0)")
        ixb = [b0["links"].index(l) for l in links]
        B0 = b0["snap"][:, ixb]
        del b0
        twin_abs = float(np.abs(B0.astype(float) - R0).max())
        say(f"  twin(0, traj0) simulated ({runs[-1]['seconds']:.0f} s), max |dq| vs no-drift {twin_abs:.1e} L/s")
    else:
        twin_abs = np.nan

    # -- pass 2: per drifting district -----------------------------------------------------------
    LAB, pre_rows, br_rows, ie_rows = [], [], [], []
    down_sign = LT["down_sign"].to_numpy(float)
    for k in names:
        P = _world(src, k, doc)
        (b_lo, b_hi), (f0, f1) = win[k]
        Dk = P.demand
        Qk = P.flows[links].to_numpy(float)[:H]
        # first hour k's demand departs from the no-drift city
        dif = np.abs(Dk[kc[k]].to_numpy(float) - D0[kc[k]].to_numpy(float)).max(axis=1)
        t_div = int(np.argmax(dif > 0)) if (dif > 0).any() else H
        pre = float(np.nanmax(_rel_dev(Qk[:t_div], R0[:t_div].astype(float)))) if t_div else 0.0
        pre_rows.append({"k": k, "t_div": t_div, "pre_drift_max_rel": pre})

        Dkk = Dk[kc[k]].to_numpy(float).sum(axis=1)
        dD = (Dkk - Dk0[k])[f0:f1]
        Q0w = R0[f0:f1].astype(float)
        orient = _orient(Q0w)
        ef = paired_effect(Qk[f0:f1], Q0w, dD, sd, f0, orient)

        # branch physics: dQ = down_sign x downstream share of k's demand change (gain units)
        if br:
            dK = (Dk[kc[k]].to_numpy(float) - D0[kc[k]].to_numpy(float))[f0:f1]
            for l in br:
                inside = np.array([c in DOWN[l] for c in kc[k]])
                exp_ = dK[:, inside].sum(axis=1) * down_sign[lix[l]] if inside.any() else np.zeros(len(dD))
                got = Qk[f0:f1, lix[l]] - Q0w[:, lix[l]]
                nd = float(dD @ dD)
                br_rows.append({"k": k, "element": l, "carries": bool(inside.any()),
                                "g_exact": float(exp_ @ dD / nd), "g_sim": float(got @ dD / nd)})

        # E1 on probe k's settled window, paired with the no-drift city's
        Ek = e1.run(Qk, Dk[e1.jcols].to_numpy(float), range(f0, f1), sd)
        e1_gate.append({"world": k, "row_dev": Ek["row_dev"], "close_dev": Ek["close_dev"]})
        C1 = _window(Ek["day"], Ek["days"], f0, f1, sd)
        C0f = _window(E0["day"], E0["days"], f0, f1, sd)
        j = names.index(k)
        m1, m0f = _renorm(C1, K), _renorm(C0f, K)
        c1, c0f = _raw(C1), _raw(C0f)
        W1 = _renorm(_weekly(Ek["day"], Ek["days"], f0, f1, sd), K)[..., j]      # n_w x L
        W0w = _renorm(_weekly(E0["day"], E0["days"], f0, f1, sd), K)[..., j]
        nw = W0w.shape[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.nanstd(W0w, axis=0, ddof=1) / np.sqrt(np.maximum(np.isfinite(W0w).sum(axis=0), 1))
            h2 = nw // 2
            half = np.abs(np.nanmean(W0w[:h2], axis=0) - np.nanmean(W0w[h2:2 * h2], axis=0)) if h2 else np.full(len(links), np.nan)
            dW = W1 - W0w
            t_p = np.nanmean(dW, axis=0) / (np.nanstd(dW, axis=0, ddof=1) / np.sqrt(np.isfinite(dW).sum(axis=0)))
        lab = pd.DataFrame({
            "element": links, "k": k, "window_lo": f0, "window_hi": f1, "orient": orient,
            "g_w": ef["g_w"], "S": ef["S"], "amp": ef["amp"], "mean_abs_dq": ef["mean_abs_dq"],
            "m_k_nodrift": m0f[:, j], "m_k_drift": m1[:, j], "dm_k": m1[:, j] - m0f[:, j],
            "c_k_nodrift": c0f[:, j], "c_k_drift": c1[:, j], "dc_k": c1[:, j] - c0f[:, j], "dm_L1": np.nansum(np.abs(m1 - m0f), axis=1),
            "dc_storage": c1[:, K] - c0f[:, K], "dm_null_se": se, "dm_null_half": half,
            "dm_t_paired": t_p, "n_weeks": nw})

        # E2: the twin with k's drift, hub held on the no-drift trajectory
        if e2:
            wb, dem, _ = el.blocked_model(wn0, Dk, r0, H)
            bk = log(el.simulate(wb, dem, H, pmin_nodes=jcols, dtype=np.float32), f"twin({k}, traj0)")
            Bk = bk["snap"][:, ixb]
            del bk
            g_tot = _gain_w(Qk[f0:f1] - Q0w, dD, sd, f0) * orient
            g_de0 = _gain_w(Bk[f0:f1].astype(float) - B0[f0:f1].astype(float), dD, sd, f0) * orient
            g_ie0 = g_tot - g_de0
            lab["g_tot"], lab["g_de0"], lab["g_ie0"] = g_tot, g_de0, g_ie0
            lab["sigma"] = sigma_share(g_de0, g_ie0, float(cfg["sigma_floor"]))
            # the twin's own error, in the units it is used in: its gain on k's drift (the hour-0 solve
            # and a handful of near-zero-flow hours differ by ~1e-3 L/s; that is not a response)
            g_err = _gain_w(B0[f0:f1].astype(float) - Q0w, dD, sd, f0)
            ie_rows.append({"k": k, "twin_max_abs_g_err": float(np.nanmax(np.abs(g_err))), "branch_max_abs_g_ie0": float(np.abs(g_ie0[[lix[l] for l in br]]).max())
                            if br else 0.0})
            del Bk
        LAB.append(lab)
        del P, Qk, Ek
        say(f"  {k}: settled {f0 // sm}-{f1 // sm} mo, drift from h {t_div} ({_time.time() - t0:.0f} s)")

    LABELS = pd.concat(LAB, ignore_index=True)
    # Nr: the response relative to the link's own district's response (size-free)
    homes = LINKS.set_index("element")["home"]
    own = LABELS[LABELS["k"] == LABELS["element"].map(homes)].set_index("element")["g_w"]
    LABELS["own"] = LABELS["k"] == LABELS["element"].map(homes)
    g_home = LABELS["element"].map(own)
    om = float(cfg["own_min"])
    LABELS["Nr"] = np.where(g_home.abs() >= om, LABELS["g_w"].abs() / g_home.abs().where(g_home.abs() >= om), np.nan)
    BR = pd.DataFrame(br_rows)
    if len(BR):
        LABELS = LABELS.merge(BR[["k", "element", "carries"]], on=["k", "element"], how="left")
    else:
        LABELS["carries"] = np.nan

    DOWNT = pd.DataFrame([{"element": l, "node": n} for l, ns in DOWN.items() for n in sorted(ns)])

    # -- gates -------------------------------------------------------------------------------------
    RUNS = pd.DataFrame(runs)
    PRE = pd.DataFrame(pre_rows)
    E1G = pd.DataFrame(e1_gate)
    bad_runs = RUNS[~RUNS["completed"] | (RUNS["unbalanced_steps"] > 0)]
    G = [
        _g("G-X0", "every run reached the horizon, converged", f"{len(bad_runs)} / {len(RUNS)} failing",
           len(bad_runs) == 0),
        _g("G-X0b", "one network model across the probe worlds", f"{len(set(models.values()))} distinct",
           len(set(models.values())) == 1),
        _g("G-X1", "splice exact (non-drifting demand equal across worlds)",
           f"{splice['max_abs_diff'].max():.1e}", splice["max_abs_diff"].max() == 0.0),
        _g("G-X2", "probe flows = no-drift city before the drift (max rel)",
           f"{PRE['pre_drift_max_rel'].max():.1e}", PRE["pre_drift_max_rel"].max() <= gth["pre_drift"]),
        _g("G-X3", "branch response = downstream demand change (gain units)",
           f"{(BR['g_exact'] - BR['g_sim']).abs().max():.1e}" if len(BR) else "no branches",
           (not len(BR)) or (BR["g_exact"] - BR["g_sim"]).abs().max() <= gth["branch_gain"]),
        _g("G-X4a", "E1 destination shares of every node sum to 1", f"{E1G['row_dev'].max():.1e}",
           E1G["row_dev"].max() <= gth["e1_row"]),
        _g("G-X4b", "E1 destination parts of every link sum to its |flow|", f"{E1G['close_dev'].max():.1e}",
           E1G["close_dev"].max() <= gth["e1_close"]),
        _g("G-X5", "E1 branch composition = downstream demand shares", f"{bdev:.1e}", bdev <= gth["e1_branch"]),
    ]
    if e2:
        IE = pd.DataFrame(ie_rows)
        G += [_g("G-X6a", "twin(0, traj0) reproduces the no-drift city (gain units; max |dq| in L/s)",
                 f"{IE['twin_max_abs_g_err'].max():.1e} ({twin_abs:.1e} L/s)",
                 IE["twin_max_abs_g_err"].max() <= gth["twin"]),
              _g("G-X6b", "every twin hour has a head reference", str(gaps), gaps == 0),
              _g("G-X6c", "no storage share on branches (|g_ie0|)", f"{IE['branch_max_abs_g_ie0'].max():.1e}",
                 IE["branch_max_abs_g_ie0"].max() <= gth["branch_ie0"])]
    GATES = pd.DataFrame(G)
    windows = {k: {"baseline": list(map(int, win[k][0])), "settled": list(map(int, win[k][1]))} for k in names}
    return {"links": LINKS, "labels": LABELS, "downstream": DOWNT, "gates": GATES, "runs": RUNS,
            "splice": splice.merge(PRE.rename(columns={"k": "district"}), on="district", how="left"),
            "manifest": {"hours": H, "steps_day": sd, "steps_month": sm, "nodrift_window": list(map(int, W0)),
                         "season_aligned": season_aligned, "windows": windows, "e2": e2,
                         "series": ["nodrift_flows.parquet"] if cfg.get("save_series", True) else []}}


def _g(gid, what, value, passed) -> dict:
    return {"gate": gid, "check": what, "value": value, "passed": bool(passed)}


def _model_signature(wn) -> dict:
    """What decides a solve: the model as wntr serialises it (patterns and demands are injected later)."""
    d = __import__("wntr").network.to_dict(wn)
    return {k: d[k] for k in ("nodes", "links", "controls", "options", "curves") if k in d}
