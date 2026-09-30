"""Stage 0 -- what this world can and cannot support, before anything is fitted.

Every downstream estimator in this workstream has a precondition that is a
property of the WORLD, not of the method. Measuring them first is what keeps
a null result attributable ("this world has no storage exchange") instead of
attributable to the estimator ("the method found no lag").

Five censuses, in the order the notebook runs them.

``phase_census``
    Months and BLOCKS per phase for a given ``block_months``, counted the way
    the pipeline cuts them: ``ceil(months / block_months)``, with the last
    block of a phase possibly shorter (`last_block_months`). The rolling
    transfer gain needs blocks inside the ramp to have anything to localise,
    so `blocks_transition` is a gate, not a statistic.

``block_budget``
    The estimation budget of a discrete chain at alphabet size k. Two
    independent floors, both reported:

        transitions needed   N >= c_cell * k^2        (c_cell ~ 10)
        plug-in MI bias      bias ~ df / (2N) nats,
                             df = (k-1)^2      for I(s_c ; s_d)
                             df = k (k-1)^2    for the TE I(s_d' ; s_c | s_d)

    `n_eff` is what the calendar key actually buys: pooling the cycle
    classes of a block gives `n_blocks * n_classes` transitions for one
    per-client chain, and that is the number the floors are read against --
    with the caveat, recorded in the note, that classes inside a month are
    not independent, so the pooled count buys ESTIMATION and not INFERENCE
    (see `chain.month_block_null`).

``occupancy``
    Windows per (group, client) at the chosen alignment. `n_min == 1` means
    the prototype IS one real window -- the privacy floor and the averaging
    floor are the same number. Below `n_min_required` the key is too fine
    for this horizon.

``channel_balance``
    Channels per client and per slot class. An unequal channel count across
    clients is a size confound wearing a regime costume: every similarity
    and every predictive gain is then partly a statement about how many
    sensors a client has. This one is an assert, not a report.

``storage_exchange``
    The lag-arm gate. For each tank, the net flux through its attached links,

        net_j(t) = sum_{l: end(l)=j} q_l(t) - sum_{l: start(l)=j} q_l(t)

    in the link sign convention (positive start->end). Three numbers:

        active_frac   share of steps with |net_j| > eps * mean total demand
        flips_day     mean sign changes of net_j per day -- storage that
                      fills and empties, rather than a one-way top-up
        throughput    sum_t |net_j(t)| / 2 / sum_t total demand, the share of
                      delivered volume that passes through storage

    Each tank gets its OWN verdict, `active = flips_day >= min_flips_day and
    throughput >= min_throughput`, and `lag_arm` is True when ANY tank is
    active. A median over tanks is the wrong summary: on ky72 two of the
    three tanks are idle (T-1, T-2: throughput 0.0003, 0.0002) while T-3
    fills and drains ~3 times a day and carries 30% of all delivered volume;
    the median reported that as "no storage exchange". A single active store
    is a lag mechanism for the districts on its side of the network, so the
    gate says WHICH tanks are active, and the next question -- which district
    pairs route through them -- is a property of the network to be measured,
    not assumed. The thresholds are arguments, printed in the register, never
    hidden constants.

``drift_ramp``
    Onset, end, ramp months, and the demand-weighted progress curve, plus
    which blocks the ramp covers -- the exact null for the onset-placement
    test in `transfer.onset_test` is uniform over evaluated blocks, so the
    count of ramp blocks IS the p-value's numerator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["phase_census", "block_budget", "occupancy", "channel_balance",
           "storage_exchange", "drift_ramp", "district_adjacency",
           "stage0", "MI_DF", "TE_DF"]


def MI_DF(k: int) -> int:
    """Degrees of freedom of the pairwise MI plug-in bias."""
    return (int(k) - 1) ** 2


def TE_DF(k: int) -> int:
    """Degrees of freedom of the conditional (transfer entropy) plug-in bias."""
    return int(k) * (int(k) - 1) ** 2


# --------------------------------------------------------------------------
# phases and blocks
# --------------------------------------------------------------------------
def phase_census(world, block_months: int = 1) -> pd.DataFrame:
    """Months and blocks per phase, with the ramp band made explicit."""
    ph = world.phases()
    n = int(world.n_months)
    first, last = world.first_switch, world.last_switch
    ramp_lo = int(first) if first is not None else ph["init"][1]
    ramp_hi = int(ph["final"][0])
    spans = {"init": ph["init"], "transition": (ramp_lo, ramp_hi),
             "final": ph["final"]}
    rows = []
    for name, (lo, hi) in spans.items():
        months = max(0, int(hi) - int(lo))
        bm = max(int(block_months), 1)
        nb = int(np.ceil(months / bm))
        rows.append({"phase": name, "m_lo": int(lo), "m_hi": int(hi) - 1,
                     "months": months, "blocks": nb,
                     "last_block_months": int(months - (nb - 1) * bm) if nb else 0})
    out = pd.DataFrame(rows)
    out.attrs["n_months"] = n
    out.attrs["first_switch"] = first
    out.attrs["last_switch"] = last
    out.attrs["ramp_months"] = int(world.ramp_months)
    return out


def block_budget(n_blocks: int, n_classes: int = 1, k: int = 4,
                 c_cell: float = 10.0) -> pd.DataFrame:
    """Estimation floors for a k-state chain given the block/class budget."""
    n_eff = int(n_blocks) * max(int(n_classes), 1)
    rows = [
        {"quantity": "transitions available (n_blocks x n_classes)",
         "value": n_eff, "floor": np.nan, "ok": np.nan},
        {"quantity": f"transitions needed for P-hat (c_cell={c_cell:g} k^2)",
         "value": n_eff, "floor": float(c_cell) * k ** 2,
         "ok": n_eff >= c_cell * k ** 2},
        {"quantity": "MI plug-in bias (nats)",
         "value": MI_DF(k) / (2 * max(n_eff, 1)), "floor": np.nan, "ok": np.nan},
        {"quantity": "TE plug-in bias (nats)",
         "value": TE_DF(k) / (2 * max(n_eff, 1)), "floor": np.nan, "ok": np.nan},
    ]
    out = pd.DataFrame(rows)
    out.attrs["k"] = int(k)
    out.attrs["n_eff"] = n_eff
    return out


def occupancy(run, n_min_required: int = 5) -> pd.DataFrame:
    """`aligned.coverage`, with the floor applied. Needs an `AlignedRun`."""
    from . import aligned as AL
    cov = AL.coverage(run.wt, run.align)
    cov["ok"] = cov["n_min"] >= int(n_min_required)
    return cov


def channel_balance(channels: pd.DataFrame) -> pd.DataFrame:
    """Channels per (client, slot_class) from `recon.channel_table`."""
    t = (channels.groupby(["client", "slot_class"]).size()
         .unstack("slot_class").fillna(0).astype(int))
    t["total"] = t.sum(axis=1)
    return t


# --------------------------------------------------------------------------
# the lag-arm gate
# --------------------------------------------------------------------------
def _tank_links(wn) -> dict[str, list[tuple[str, int]]]:
    """{tank: [(link, sign), ...]}, sign +1 when the link ENDS at the tank."""
    out: dict[str, list[tuple[str, int]]] = {}
    for t in list(wn.tank_name_list):
        entries = []
        for name in wn.link_name_list:
            lk = wn.get_link(name)
            if getattr(lk, "end_node_name", None) == t:
                entries.append((name, +1))
            elif getattr(lk, "start_node_name", None) == t:
                entries.append((name, -1))
        out[t] = entries
    return out


def storage_exchange(world, eps_frac: float = 0.01, min_flips_day: float = 1.0,
                     min_throughput: float = 0.02) -> dict:
    """Is there ACTIVE storage exchange on this network?

    Returns `{"per_tank": DataFrame, "verdict": Series, "available": bool,
    "reason": str}`. `available=False` (no tanks, no `wn_variant.pkl`, no
    flow columns) leaves the lag arm gated rather than guessing.
    """
    out = {"per_tank": pd.DataFrame(), "available": False, "reason": "",
           "verdict": pd.Series(dtype=float)}
    try:
        wn = world.wn()
    except Exception as exc:                       # no pickle, no wntr
        out["reason"] = f"network object unavailable: {type(exc).__name__}: {exc}"
        return out
    tanks = list(getattr(wn, "tank_name_list", []) or [])
    if not tanks:
        out["reason"] = "network has no tanks"
        return out

    flows = world.flows
    links = _tank_links(wn)
    dem = world.district_demand
    total = dem[[c for c in dem.columns if c != "month"]].sum(axis=1).to_numpy(float)
    n = min(len(flows), len(total))
    total = total[:n]
    eps = float(eps_frac) * float(np.mean(total))
    steps_day = int(world.steps_day)

    rows = []
    for t in tanks:
        cols = [(l, s) for l, s in links[t] if l in flows.columns]
        if not cols:
            continue
        net = np.zeros(n, float)
        for l, s in cols:
            net += s * flows[l].to_numpy(float)[:n]
        sgn = np.sign(net)
        sgn = sgn[np.abs(net) > eps]
        flips = int(np.sum(np.diff(sgn) != 0)) if len(sgn) > 1 else 0
        days = max(n / steps_day, 1.0)
        rows.append({
            "tank": t, "n_links": len(cols),
            "active_frac": float(np.mean(np.abs(net) > eps)),
            "flips_day": flips / days,
            "throughput": float(np.sum(np.abs(net)) / 2.0
                                / max(np.sum(total), 1e-12)),
            "net_mean": float(np.mean(net)), "net_std": float(np.std(net))})
    per = pd.DataFrame(rows)
    if not len(per):
        out["reason"] = "no flow columns for any tank link"
        return out

    per["active"] = ((per["flips_day"] >= float(min_flips_day))
                     & (per["throughput"] >= float(min_throughput)))
    verdict = pd.Series({
        "n_tanks": int(len(per)),
        "n_active": int(per["active"].sum()),
        "active_tanks": per.loc[per["active"], "tank"].tolist(),
        "max_throughput": float(per["throughput"].max()),
        "min_flips_day": float(min_flips_day),
        "min_throughput": float(min_throughput)})
    verdict["lag_arm"] = bool(verdict["n_active"] > 0)
    out.update(per_tank=per, verdict=verdict, available=True,
               reason="measured")
    return out


def drift_ramp(world, block_months: int = 1) -> dict:
    """Onset/end, progress curve, and which blocks the ramp covers."""
    prog = world.drift_progress()
    first, last = world.first_switch, world.last_switch
    ph = world.phases()
    lo = int(first) if first is not None else int(ph["init"][1])
    hi = int(ph["final"][0])
    bm = max(int(block_months), 1)
    blocks = np.arange(int(np.ceil(world.n_months / bm)))
    b_lo, b_hi = blocks * bm, blocks * bm + bm - 1
    in_ramp = (b_hi >= lo) & (b_lo < hi)
    return {"first_switch": first, "last_switch": last,
            "ramp_months": int(world.ramp_months),
            "ramp_span": (lo, hi), "progress": prog,
            "ramp_blocks": [int(b) for b in blocks[in_ramp]],
            "n_blocks": int(len(blocks))}


# --------------------------------------------------------------------------
# adjacency (used by partialdep.structural_contrast as EVALUATION only)
# --------------------------------------------------------------------------
def district_adjacency(world) -> tuple[pd.DataFrame, str]:
    """(frame of district pairs with `adjacent`, source used).

    Two routes, in order: the world's own `partition.yml` manifest when it
    records boundary links, else the network object (a link whose endpoints
    sit in different districts is a boundary). Returns an empty frame and a
    reason when neither is available -- this is ground truth for SCORING, so
    it is never guessed.
    """
    names = list(world.client_names)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    try:
        info = world.partition_info() or {}
        key = next((k for k in info
                    if "boundary" in str(k).lower() or "cut" in str(k).lower()),
                   None)
        if key is not None:
            recorded = info[key]
            edges = set()
            if isinstance(recorded, dict):
                for k2, v in recorded.items():
                    if isinstance(v, (int, float)) and float(v) > 0 and "|" in str(k2):
                        a, b = str(k2).split("|")[:2]
                        edges.add(tuple(sorted((a.strip(), b.strip()))))
            elif isinstance(recorded, list):
                for e in recorded:
                    if isinstance(e, (list, tuple)) and len(e) >= 2:
                        edges.add(tuple(sorted((str(e[0]), str(e[1])))))
            if edges:
                df = pd.DataFrame([{"client_a": a, "client_b": b,
                                    "adjacent": tuple(sorted((a, b))) in edges}
                                   for a, b in pairs])
                return df, f"partition_info[{key!r}]"
    except Exception:
        pass

    try:
        wn = world.wn()
        owner = {n: d for d, nodes in world.districts.items() for n in nodes}
        edges = set()
        for name in wn.link_name_list:
            lk = wn.get_link(name)
            a = owner.get(str(getattr(lk, "start_node_name", "")))
            b = owner.get(str(getattr(lk, "end_node_name", "")))
            if a and b and a != b:
                edges.add(tuple(sorted((a, b))))
        df = pd.DataFrame([{"client_a": a, "client_b": b,
                            "adjacent": tuple(sorted((a, b))) in edges}
                           for a, b in pairs])
        return df, "network links (cross-district)"
    except Exception as exc:
        return (pd.DataFrame(columns=["client_a", "client_b", "adjacent"]),
                f"unavailable: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# the whole stage
# --------------------------------------------------------------------------
def stage0(world, channels: pd.DataFrame | None = None, block_months: int = 1,
           n_classes: int = 1, k: int = 4, min_blocks_transition: int = 8,
           reg=None, **storage_kw) -> dict:
    """Run every census and file the verdicts. `reg` is a `checks.Register`."""
    from .checks import Register
    reg = Register("stage0") if reg is None else reg
    ph = phase_census(world, block_months)
    ramp = drift_ramp(world, block_months)
    tr = int(ph.loc[ph["phase"] == "transition", "blocks"].iloc[0])

    reg.report("stage0", "world", world.tag,
               f"{world.network} / {world.districting_method} / "
               f"{world.placement_source}")
    reg.report("stage0", "months per phase",
               dict(zip(ph["phase"], ph["months"])))
    reg.gate("stage0", f"blocks in transition (block_months={block_months})",
             tr, tr >= int(min_blocks_transition),
             f"rolling G needs >= {min_blocks_transition}")
    reg.report("stage0", "ramp blocks", ramp["ramp_blocks"],
               f"span months {ramp['ramp_span']}")

    bud = block_budget(int(ph["blocks"].sum()), n_classes, k)
    ok = bool(bud.loc[bud["ok"].notna(), "ok"].all())
    reg.gate("stage0", f"chain budget at k={k}", bud.attrs["n_eff"], ok,
             "blocks x classes against c_cell k^2; classes are not "
             "independent within a month")

    if channels is not None:
        cb = channel_balance(channels)
        bal = bool(cb["total"].nunique() == 1)
        reg.expect("stage0", "channels per client equal", cb["total"].to_dict(),
                   bal, "unequal D_m is a size confound")

    st = storage_exchange(world, **storage_kw)
    if st["available"]:
        reg.gate("stage0", "active storage exchange (lag arm)",
                 {k2: round(v, 4) if isinstance(v, float) else v
                  for k2, v in st["verdict"].items()},
                 bool(st["verdict"]["lag_arm"]),
                 "open = at least one tank fills and drains; which district "
                 "pairs route through it is the next measurement")
    else:
        reg.report("stage0", "active storage exchange (lag arm)", "unavailable",
                   st["reason"] + " -- lag arm stays closed")

    return {"phases": ph, "budget": bud, "ramp": ramp, "storage": st,
            "channels": None if channels is None else channel_balance(channels),
            "reg": reg,
            "lag_arm": bool(st["available"] and st["verdict"]["lag_arm"])}
