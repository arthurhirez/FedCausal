"""mixture_probe.py -- what fraction of which district is a sensor reading?

The idea
--------
The drift sweep is already an excitation experiment. Each of its worlds moves
ONE district and holds everything else fixed, so stacking the worlds of a
network gives a design matrix with exactly one active column per world. That
makes the mixture identifiable at district level from the real, settled
simulation -- no probe world, no synthetic perturbation, no collinearity.

This is deliberately NOT the earlier Jacobian probe. Three differences, and
each is the reason that one did not cluster clients:

* the excitation is a real land-use conversion at realistic magnitude, not an
  infinitesimal nudge;
* it is measured after the new regime has SETTLED, at the actual operating
  point, not on a base topology;
* it is taken on the z-scored weekly profile, so a pure level change scores
  zero and what is measured is the change of temporal signature -- the part
  that survives the FL scaler. The Jacobian could not separate the two.

The estimator
-------------
For sensor `s` in the world where district `k` drifted::

    dz_s      = z(settled weekly profile) - z(init weekly profile)
    g[s, k]   = <dz_s, dz_d(k)> / ||dz_d(k)||^2

`g` is the OLS coefficient of the sensor's shape change on that district's
demand shape change: "how much of district k's signature reached this gauge".
Stacked over k and normalised it is a simplex, so `0.75 home / 0.25 District_X`
is a literal reading rather than a metaphor.

The noise floor is a SPLIT-HALF NULL taken inside the init phase: the same
`||dz||` between two disjoint halves of a stretch where nothing drifted. A
gauge whose response does not clear it is not a mixed gauge, it is an
unresponsive one -- and that is a different physical object (transit flow, or
storage exchange) rather than a noisy version of the same one.

Tiers
-----
    unusable      response does not clear the split-half null
    core          clears it, purity >= `core_purity`
    transition    clears it, purity below that -- a quantifiable mixture

Contiguity of a core set is MEASURED, never assumed. Hydraulic influence
follows the supply path, not Euclidean distance, so the mixture field has no
reason a priori to be spatially smooth.
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

import signal_probe as sp

__all__ = ["settle_scan", "settle_month", "settle_report", "settled_window",
           "split_half_null", "horizon_check",
           "world_response", "build_mixture", "remix", "classify", "mixture_diagnostics",
           "tier_sensitivity", "purity_histogram", "dependence_matrix",
           "dependence_spread", "zones", "elasticity", "dependence",
           "dependence_summary", "asymmetry", "estimator_checks",
           "beta_linearity", "response_matrix", "null_vs_storage",
           "plot_null_vs_storage", "plot_matrix",
           "core_connectivity", "beta_stability", "load_cells",
           "plot_settle", "plot_purity", "plot_dependence", "plot_mixture_bars",
           "plot_tier_map"]

EPS = 1e-12


# ==========================================================================
# 1. settling -- when is the new regime actually stable?
# ==========================================================================
def settle_scan(P: sp.Probe, ref_months: int = 3) -> pd.DataFrame:
    """Per district and month: correlation of that month's weekly demand
    profile with the terminal one.

    The observation this exists to make measurable: adherence between the best
    gauges and the district's demand pattern only appears a few months AFTER
    the last node switches. `World.phases()` opens `final` at
    `last_switch + ramp_months`, which is too early if that holds -- the
    profile is still moving there, so an init/final contrast mixes the settled
    regime with the tail of the transition.

    The reference is the MEAN profile over the last `ref_months`, not the last
    month alone. A single month carries its own weather draw and month-level
    volume wobble; correlating against it makes the target's noise part of the
    criterion and the answer then depends on which month the horizon happened
    to end on.

    On a tankless network settling should be immediate. If it is not immediate
    here, what is settling is storage re-entering a duty cycle -- the same
    mechanism the lag search has been after.
    """
    sd, sm = P.steps_day, P.steps_month
    n = int(P.time["n_months"])
    if sm < 7 * sd:
        raise ValueError("a month shorter than a week has no weekly profile; "
                         "raise days_per_month to at least 7")
    dd = P.district_demand

    def prof(col, m):
        lo = m * sm
        return sp.zscore(sp.weekly_profile(dd[col].to_numpy()[lo:lo + sm], sd,
                                           step0=lo))

    ref = {d: sp.zscore(np.mean([prof(d, m) for m in range(n - ref_months, n)],
                                axis=0)) for d in dd.columns}
    rows = []
    for m in range(n):
        for d in dd.columns:
            w = prof(d, m)
            rows.append({"month": m, "district": d,
                         "r_to_terminal": float(w @ ref[d] / len(w))})
    return pd.DataFrame(rows)


def _threshold(r: np.ndarray, ref_months: int, slack: float) -> float:
    """Ceiling minus a slack that ADAPTS to the world's own reproducibility.

    A flat slack is wrong because the ceiling is network- and horizon-specific:
    on a large quiet district the month-to-month correlation sits at 1.000 and
    0.01 is generous, while on a small district with a real weather draw it
    sits at 0.95 and 0.01 is inside the noise. So the slack is at least twice
    the spread of the reference months themselves.
    """
    tail = r[-ref_months:]
    return float(np.median(tail)) - max(slack, 2.0 * float(np.std(tail)))


def settle_month(P: sp.Probe, ref_months: int = 3, slack: float = 0.01,
                 scan: pd.DataFrame | None = None) -> dict:
    """First month from which a district's profile STAYS at its terminal shape.

    The threshold is self-calibrating rather than a fixed number. A month's
    correlation to the terminal reference cannot reach 1.0 even when the
    regime is stationary -- weather, the month-level volume draw and the crowd
    residual all put a ceiling on it -- and that ceiling is network- and
    horizon-specific. So the ceiling is MEASURED as the median correlation
    over the reference months themselves, and "settled" means within `slack`
    of it. A fixed 0.99 instead reports "never settled" on any world noisy
    enough to be realistic.

    "Stays" is load-bearing: a single lucky month mid-transition must not
    count, so the test is run from the end backwards.
    """
    scan = settle_scan(P, ref_months) if scan is None else scan
    out = {}
    for d, g in scan.groupby("district"):
        g = g.sort_values("month")
        r = g["r_to_terminal"].to_numpy()
        ok = r >= _threshold(r, ref_months, slack)
        stay = np.where(np.cumprod(ok[::-1])[::-1] == 1)[0]
        out[d] = int(g["month"].to_numpy()[stay[0]]) if len(stay) else None
    return out


def settle_report(P: sp.Probe, ref_months: int = 3, slack: float = 0.01) -> pd.DataFrame:
    """Settle month per district, with the reproducibility ceiling alongside.

    The ceiling is the diagnostic that says whether `slack` is sensible: if a
    stationary district's own month-to-month correlation sits at 0.97, a slack
    of 0.01 is measuring noise, not settling.
    """
    scan = settle_scan(P, ref_months)
    s = settle_month(P, ref_months, slack, scan=scan)
    tgt = P.drift.get("tgt_district")
    rows = []
    for d, g in scan.groupby("district"):
        r = g.sort_values("month")["r_to_terminal"].to_numpy()
        rows.append({"district": d, "drifted": d == tgt,
                     "settle_month": s[d],
                     "ceiling": float(np.median(r[-ref_months:])),
                     "threshold": _threshold(r, ref_months, slack),
                     "r_min": float(r.min()),
                     "r_at_first_final": float(r[min(P.phases()["final"][0],
                                                     len(r) - 1)])})
    return pd.DataFrame(rows).round(4)


def settled_window(P: sp.Probe, ref_months: int = 3, slack: float = 0.01,
                   pad: int = 1, min_months: int = 3) -> tuple[int, int]:
    """Step window of the SETTLED post-drift regime, as (lo, hi).

    Driven by the DRIFTING district. The others are stationary, so a late
    settle month from one of them is a noise draw and would otherwise throw
    the window away.
    """
    s = settle_month(P, ref_months, slack)
    n = int(P.time["n_months"])
    tgt = P.drift.get("tgt_district")
    base = s.get(tgt)
    if base is None:
        known = [v for v in s.values() if v is not None]
        base = max(known) if known else P.phases()["final"][0]
    start = int(base) + pad
    if n - start < min_months:
        raise ValueError(
            f"settled window is {n - start} month(s) (settle at {s}); extend "
            f"the horizon to at least {base + pad + min_months} months, or the "
            "post-drift profile is still moving when it is measured")
    return start * P.steps_month, n * P.steps_month


def baseline_window(P: sp.Probe) -> tuple[int, int]:
    """The drift-free reference: the `init` phase, in steps."""
    return P.month_slice("init")


# ==========================================================================
# 2. one world -> response vectors
# ==========================================================================
def _profile(P: sp.Probe, cand: pd.DataFrame, lo: int, hi: int):
    """Aligned sensor and district weekly profiles over an arbitrary window.

    Delegates to `sp.aligned_profile` / `sp.district_profile`, which memoise
    on (window, candidate order). The arithmetic is the one that used to live
    here, unchanged -- `world_response` and the two null estimators just stop
    recomputing the same three windows ten times per world.
    """
    S, sign = sp.aligned_profile(P, cand, lo, hi)
    D = sp.district_profile(P, lo, hi)
    return S, D, sign


def _split_half_delta(P: sp.Probe, cand: pd.DataFrame,
                      windows: tuple | None = None,
                      space: str = "shape") -> np.ndarray:
    """The split-half difference VECTOR, scaled like the measurement.

    `split_half_null` returns its norm; the projection-based estimator needs
    the vector itself so the noise can be projected onto the same direction as
    the signal.
    """
    lo, hi = baseline_window(P)
    sd = P.steps_day
    mid = lo + ((hi - lo) // (2 * sd)) * sd
    A, _, _ = _profile(P, cand, lo, mid)
    B, _, _ = _profile(P, cand, mid, hi)
    dN = (sp.zscore(B) - sp.zscore(A)) if space == "shape" else (B - A)
    if windows is None:
        return dN
    (a0, a1), (b0, b1) = windows
    n_h, n_a, n_b = (mid - lo) / sd, (a1 - a0) / sd, (b1 - b0) / sd
    if min(n_h, n_a, n_b) <= 0:
        return dN
    return dN * float(np.sqrt((1 / n_a + 1 / n_b) / (2 / n_h)))


def split_half_null(P: sp.Probe, cand: pd.DataFrame,
                    windows: tuple | None = None) -> np.ndarray:
    """Per-sensor noise floor for ||dz||, from two halves of the init phase.

    Nothing drifts during `init`, so whatever `||dz||` comes out of splitting
    it is the shape wobble the world produces on its own -- weather, the
    month-level volume draw, the crowd residual. A response below this is not
    a small response; it is no response.

    SAMPLE-SIZE MATCHING, and it is not optional. The null compares two
    half-init windows; the measurement compares the whole init phase against a
    settled window many times longer. A weekly profile's sampling error falls
    as 1/sqrt(days), so an unmatched null is inflated by the ratio of those
    lengths and would condemn most of the network as unresponsive purely
    because the null was estimated on less data than the signal. `windows`
    carries the two windows actually being compared; the null is rescaled by
    sqrt((1/n_a + 1/n_b) / (2/n_half)) in days.
    """
    lo, hi = baseline_window(P)
    sd = P.steps_day
    mid = lo + ((hi - lo) // (2 * sd)) * sd
    A, _, _ = _profile(P, cand, lo, mid)
    B, _, _ = _profile(P, cand, mid, hi)
    raw = np.linalg.norm(sp.zscore(B) - sp.zscore(A), axis=1)
    if windows is None:
        return raw
    (a0, a1), (b0, b1) = windows
    n_h = (mid - lo) / sd
    n_a, n_b = (a1 - a0) / sd, (b1 - b0) / sd
    if min(n_h, n_a, n_b) <= 0:
        return raw
    return raw * float(np.sqrt((1 / n_a + 1 / n_b) / (2 / n_h)))


def world_response(P: sp.Probe, cand: pd.DataFrame, slack: float = 0.01,
                   pad: int = 1, ref_months: int = 3) -> pd.DataFrame:
    """One world -> one column of the gain matrix.

    Returns per sensor: the regression coefficient of its shape change on the
    drifting district's demand shape change, the residual left over, and the
    same quantities for level so the two can be told apart.
    """
    tgt = P.drift["tgt_district"]
    lo0, hi0 = baseline_window(P)
    lo1, hi1 = settled_window(P, ref_months=ref_months, slack=slack, pad=pad)

    S0, D0, _ = _profile(P, cand, lo0, hi0)
    S1, D1, _ = _profile(P, cand, lo1, hi1)
    names = list(P.district_demand.columns)
    k = names.index(tgt)

    dS = sp.zscore(S1) - sp.zscore(S0)
    dD = sp.zscore(D1)[k] - sp.zscore(D0)[k]
    excite = float(np.linalg.norm(dD))            # size of THIS excitation
    u = dD / excite if excite > EPS else dD * 0.0  # its unit direction
    proj = dS @ u                                  # response along that direction
    g = proj / excite if excite > EPS else np.zeros(len(dS))
    resid = np.linalg.norm(dS - np.outer(proj, u), axis=1)

    # The noise floor FOR THE PROJECTION, not for the norm. Projecting the
    # split-half difference onto the same unit direction gives the error this
    # particular measurement carries; it is much smaller than ||dz||, because a
    # projection averages the noise over the profile instead of accumulating
    # it. Using the norm-null to gate a projection is what forced `null_factor`
    # so high that mixtures collapsed to one-hot.
    dN = _split_half_delta(P, cand, windows=((lo0, hi0), (lo1, hi1)))
    nu = np.abs(dN @ u)

    # The SAME estimator in native units (mca, L/s), not on z-scored profiles.
    # Needed because the two channels answer to different excitations: Graeme's
    # pressure elasticity tracked the volume level factor 9.19**beta almost
    # exactly (slope 0.422 / 0.613 / 0.785 / 1.000 / 1.126 / 1.418 against a
    # predicted 0.46 / 0.64 / 0.81 / 1.00 / 1.12 / 1.36), i.e. dividing by the
    # SHAPE norm left a beta-dependent residue. In native units the excitation
    # is the district's demand-profile change in L/s, which carries level and
    # shape together, so E is the linearised operator itself and should be
    # excitation-invariant if the network responds linearly.
    dSn, dDn = S1 - S0, D1[k] - D0[k]
    exc_n = float(np.linalg.norm(dDn))
    un = dDn / exc_n if exc_n > EPS else dDn * 0.0
    proj_n = dSn @ un
    dNn = _split_half_delta(P, cand, windows=((lo0, hi0), (lo1, hi1)),
                            space="native")
    nu_n = np.abs(dNn @ un)

    # level, kept separate on purpose: a gauge can move a lot in volume and
    # not at all in signature, and only the signature survives the FL scaler.
    lvl0, lvl1 = S0.mean(axis=1), S1.mean(axis=1)
    dlv = np.divide(lvl1 - lvl0, np.abs(lvl0), out=np.zeros(len(lvl0)),
                    where=np.abs(lvl0) > EPS)
    dD_lvl = float(D1[k].mean() / D0[k].mean() - 1.0) if abs(D0[k].mean()) > EPS else np.nan

    return pd.DataFrame({
        "id": cand["id"].to_numpy(), "kind": cand["kind"].to_numpy(),
        "district": cand["district"].to_numpy(), "role": cand["role"].to_numpy(),
        "element": cand["element"].to_numpy(),
        "drift_district": tgt, "g": g, "proj": proj, "nu": nu,
        "excite": excite, "proj_native": proj_n, "nu_native": nu_n,
        "excite_native": exc_n, "dz_norm": np.linalg.norm(dS, axis=1),
        "resid": resid,
        "null": split_half_null(P, cand, windows=((lo0, hi0), (lo1, hi1))),
        "null_raw": split_half_null(P, cand),
        "d_level": dlv, "d_level_district": dD_lvl,
        "settle_start_month": lo1 // P.steps_month,
    })


# ==========================================================================
# 3. stack the worlds -> mixture
# ==========================================================================
def horizon_check(probes: dict, ref_months: int = 3, slack: float = 0.01,
                 pad: int = 1, min_months: int = 3) -> pd.DataFrame:
    """Pre-flight: does every world have a long enough SETTLED stretch?

    Run this before `build_mixture`. A world whose settled window is short
    does not fail loudly on its own -- it quietly measures a profile that is
    still moving -- so the horizon is checked once, up front, and the table
    says how many months each world would need.
    """
    rows = []
    for d, P in probes.items():
        n = int(P.time["n_months"])
        sm = settle_month(P, ref_months, slack)
        tgt = P.drift.get("tgt_district", d)
        base = sm.get(tgt)
        base = int(base) if base is not None else P.phases()["final"][0]
        start = base + pad
        rows.append({"drift_district": d, "n_months": n,
                     "last_switch": (int(P.schedule["drift_month"].max())
                                     if len(P.schedule) else None),
                     "settle_month": base,
                     "settled_from": start, "settled_months": n - start,
                     "ok": (n - start) >= min_months,
                     "need_n_months": base + pad + min_months})
    d = pd.DataFrame(rows)
    if not d["ok"].all():
        print("HORIZON TOO SHORT for "
              f"{list(d.loc[~d['ok'], 'drift_district'])} -- rerun the sweep "
              f"with n_months >= {int(d['need_n_months'].max())}")
    return d


def load_cells(network: str, beta: float, cfg, root, out_root):
    """The sweep's worlds for one network at one beta, as Probes.

    Reads the content-addressed cache the sweep already wrote; nothing is
    re-solved here.
    """
    import landuse_sweep as ls
    from worlds import load_world
    out = {}
    for wc in ls.cells(cfg, root):
        if wc.network != network or float(wc.beta) != float(beta):
            continue
        d = pathlib.Path(out_root) / __import__("landuse_world").sim_hash(wc)
        if not (d / "manifest.json").exists():
            print(f"  missing world for {wc.tgt_district} beta={beta} ({d.name})")
            continue
        out[wc.tgt_district] = sp.from_lab_world(load_world(d))
    return out


def build_mixture(probes: dict, slack: float = 0.01, pad: int = 1,
                  ref_months: int = 3, null_factor: float = 3.0,
                  core_purity: float = 0.85, foreign_max: float = 0.15,
                  verbose: bool = True) -> dict:
    """Stack one world per drifting district -> the mixture table.

    `probes` is {drifting district: Probe}. Every world must share the
    network, the initial map and the seed, so their `init` phases are the same
    world and the differences are attributable to the drift alone.

    Returns {"gains", "mixture", "G", "districts", "cand"}.
    """
    districts = sorted(probes)
    P0 = probes[districts[0]]

    # ONE candidate frame for every world, not one per world.
    #
    # `candidates` assigns a BOUNDARY link to the district it delivers into,
    # by the sign of its mean flow -- and a drift large enough to reverse a
    # crossing changes that assignment, which changes the frame's sort order
    # (it sorts by kind, district, id). Comparing ordered id lists then fails
    # on exactly the networks where the crossings are pipes: KY7 has nine, and
    # an industrial conversion reverses some of them.
    #
    # Direction-dependent home districts are wrong for a cross-world stack
    # anyway: a gauge must mean the same thing in every column of the gain
    # matrix. So the reference world's frame is used throughout, and the check
    # is on the SET of ids -- order is irrelevant, since `_series_matrix`
    # looks columns up by name.
    ref = sp.candidates(P0)
    common = set(ref["id"])
    for d in districts[1:]:
        common &= set(sp.candidates(probes[d])["id"])
    dropped = len(ref) - len(common)
    if dropped:
        print(f"  {dropped} candidate(s) not present in every world "
              "(demand-carrying status differs) -- dropped from the stack")
    cand = ref[ref["id"].isin(common)].reset_index(drop=True)

    frames = []
    for d in districts:
        P = probes[d]
        frames.append(world_response(P, cand, slack=slack, pad=pad,
                                     ref_months=ref_months))
        if verbose:
            print(f"  {d}: settled from month "
                  f"{int(frames[-1]['settle_start_month'].iat[0])}", flush=True)
    gains = pd.concat(frames, ignore_index=True)

    G = gains.pivot(index="id", columns="drift_district", values="g")
    G = G.reindex(columns=districts)
    DZ = gains.pivot(index="id", columns="drift_district", values="dz_norm") \
        .reindex(columns=districts)
    PROJ = gains.pivot(index="id", columns="drift_district", values="proj") \
        .reindex(columns=districts)
    NU = gains.pivot(index="id", columns="drift_district", values="nu") \
        .reindex(columns=districts)
    EXC = gains.groupby("drift_district")["excite"].first().reindex(districts)
    PROJ_N = gains.pivot(index="id", columns="drift_district",
                         values="proj_native").reindex(columns=districts)
    NU_N = gains.pivot(index="id", columns="drift_district",
                       values="nu_native").reindex(columns=districts)
    EXC_N = gains.groupby("drift_district")["excite_native"].first() \
        .reindex(districts)
    NULL = gains.groupby("id")["null"].max().reindex(G.index)
    # A gauge whose init-phase series is constant (a pump on a fixed duty, a
    # closed valve) has a null of ~0, and dividing by it produced SNRs of
    # 1e13. Floor the null at a small fraction of the population median and
    # flag those rows: their SNR is not a measurement.
    floor = 1e-3 * float(np.nanmedian(NULL.to_numpy()))
    degenerate = NULL.to_numpy() < floor
    NULL = pd.Series(np.maximum(NULL.to_numpy(), floor), index=NULL.index,
                     name="null")
    base = {"degenerate": pd.Series(degenerate, index=NULL.index), "gains": gains,
            "PROJ": PROJ, "NU": NU, "EXC": EXC,
            "PROJ_native": PROJ_N, "NU_native": NU_N, "EXC_native": EXC_N,
            "G": G, "DZ": DZ, "null": NULL,
            "districts": districts, "cand": cand}
    return remix(base, null_factor=null_factor, core_purity=core_purity,
                 foreign_max=foreign_max)


def remix(base: dict, null_factor: float = 3.0, core_purity: float = 0.85,
          foreign_max: float = 0.15) -> dict:
    """Recompute the mixture from stored gains -- no re-profiling, no solving.

    Exists so the two thresholds can be swept cheaply. They are choices, and a
    result that moves a lot between `null_factor` 2 and 4 is a result about
    the threshold rather than about the network.
    """
    districts, G, DZ = base["districts"], base["G"], base["DZ"]
    NULL, cand = base["null"], base["cand"]

    # Only responses that clear the null carry mixture mass. A negative gain is
    # a real anti-phase response and is recorded separately rather than folded
    # into the simplex, where it would have no meaning.
    live = DZ.to_numpy() > (null_factor * NULL.to_numpy()[:, None])
    pos = np.where(live, np.clip(G.to_numpy(), 0, None), 0.0)
    neg = np.where(live, np.clip(G.to_numpy(), None, 0), 0.0)
    tot = pos.sum(axis=1)
    M = np.divide(pos, tot[:, None], out=np.zeros_like(pos), where=tot[:, None] > EPS)
    # `n_live` counts responses that cleared the null; `n_mass` those that also
    # point the same way as the district's own change. A gauge whose only live
    # response is ANTI-phase carries no mixture mass, and calling it a
    # transition gauge would put a zero row on the simplex.
    n_mass = (pos > 0).sum(axis=1)

    home = cand.set_index("id")["district"].reindex(G.index).to_numpy()
    hidx = np.array([districts.index(h) if h in districts else -1 for h in home])
    purity = np.array([M[i, hidx[i]] if hidx[i] >= 0 else np.nan
                       for i in range(len(M))])
    hhi = (M ** 2).sum(axis=1)
    ent = -(np.where(M > 0, M * np.log(np.where(M > 0, M, 1)), 0)).sum(axis=1)

    second = np.full(len(M), "", dtype=object)
    second_w = np.zeros(len(M))
    for i in range(len(M)):
        for j in np.argsort(-M[i]):
            if j != hidx[i] and M[i, j] > 0:
                second[i], second_w[i] = districts[j], M[i, j]
                break

    mix = pd.DataFrame(M, index=G.index, columns=[f"w_{d}" for d in districts])
    mix.insert(0, "home", home)
    for col in ("kind", "role", "element"):
        mix[col] = cand.set_index("id")[col].reindex(G.index)
    mix["n_live"] = live.sum(axis=1)
    mix["n_mass"] = n_mass
    mix["mass"] = tot
    mix["purity"] = purity
    mix["hhi"] = hhi
    mix["entropy"] = ent
    mix["second"] = second
    mix["w_second"] = second_w
    mix["neg_mass"] = -neg.sum(axis=1)
    mix["degenerate_null"] = base["degenerate"].reindex(G.index).to_numpy()
    mix["snr_home"] = np.array([
        DZ.to_numpy()[i, hidx[i]] / max(NULL.to_numpy()[i], EPS) if hidx[i] >= 0
        else np.nan for i in range(len(M))])
    mix = classify(mix.reset_index(), core_purity=core_purity,
                   foreign_max=foreign_max)
    out = dict(base)
    out.update({"mixture": mix, "null_factor": null_factor,
                "core_purity": core_purity, "foreign_max": foreign_max})
    return out


def mixture_diagnostics(res: dict) -> pd.DataFrame:
    """Is the mixture informative, or is the threshold doing all the work?

    The number to look at is `n_live<=1`. If nearly every gauge cleared the
    null for exactly one district, the "mixture" is 1.0 by construction and
    says nothing -- either the null is set too high, or the worlds really are
    decoupled enough that no gauge sees two districts. Those are different
    conclusions and this table is what separates them, with the null-factor
    sweep in `tier_sensitivity` as the check.
    """
    m = res["mixture"]
    rows = []
    for kind, g in m.groupby("kind"):
        rows.append({
            "kind": kind, "n": len(g),
            "unusable_%": 100 * float((g["tier"] == "unusable").mean()),
            "n_live<=1_%": 100 * float((g["n_live"] <= 1).mean()),
            "n_live_mean": float(g["n_live"].mean()),
            "purity_median": float(g.loc[g["tier"] != "unusable", "purity"].median()),
            "snr_median": float(g["snr_home"].median()),
            "neg_mass_%": 100 * float((g["neg_mass"] > 0).mean()),
        })
    return pd.DataFrame(rows).round(3)


def tier_sensitivity(base: dict, factors=(1.5, 2.0, 3.0, 5.0),
                     purities=(0.85,), foreign_max: float = 0.15) -> pd.DataFrame:
    """How the tiers move with the two thresholds. Cheap: no re-solving.

    The two axes are not equally informative and should not be read the same
    way. `null_factor` decides which gauges enter the simplex at all, so it
    can create or destroy mixtures -- a real measurement decision.
    `core_purity` only RE-LABELS a purity distribution that is already fixed,
    so sweeping it tells you where the mass of that distribution sits, which
    `purity_histogram` shows directly and better. Sweep it to check that a
    reported core count is not balanced on a cliff edge; do not read it as
    evidence about the network.
    """
    rows = []
    for f in factors:
        for cp in purities:
            m = remix(base, null_factor=f, core_purity=cp,
                      foreign_max=foreign_max)["mixture"]
            for kind, g in m.groupby("kind"):
                rows.append({"null_factor": f, "core_purity": cp, "kind": kind,
                             **{t: int((g["tier"] == t).sum())
                                for t in ("core", "transition", "foreign",
                                          "unusable")},
                             "n_live_mean": round(float(g["n_live"].mean()), 2)})
    return pd.DataFrame(rows)


def purity_histogram(res: dict, bins=(0, .15, .3, .5, .7, .85, .95, 1.001)) -> pd.DataFrame:
    """Where the purity mass actually sits, for gauges that cleared the null.

    This is the table a `core_purity` sweep is a roundabout way of asking for.
    A distribution piled at 1.0 with nothing between 0.3 and 0.85 means the
    threshold is irrelevant and no gauge carries a readable MIXTURE; mass in
    the middle is the population that can support a quantified dependence
    statement.
    """
    m = res["mixture"]
    m = m[m["tier"] != "unusable"]
    out = (m.assign(bin=pd.cut(m["purity"], bins=list(bins), right=False))
           .groupby(["kind", "bin"], observed=False).size().rename("n")
           .reset_index())
    out["share"] = out.groupby("kind")["n"].transform(lambda x: x / max(x.sum(), 1))
    return out.round(3)


def classify(mix: pd.DataFrame, core_purity: float = 0.85,
             foreign_max: float = 0.15) -> pd.DataFrame:
    """core | transition | foreign | unusable.

    `unusable` is NOT "noisy": it is a gauge whose signature did not move for
    ANY district's drift -- dominated by transit flow or storage exchange, a
    different physical object from a boundary gauge reading two districts at
    once.

    `foreign` was added after the fact, from D-Town: several links INSIDE
    District_C responded strongly and almost entirely to District_B's drift
    (purity 0.00, w_second 1.00, with anti-phase mass against their own
    district). Those are transit pipes whose flow reverses when the local
    district draws more. Calling them "transition" hides the most actionable
    category there is -- a place a district's own meter must never go.
    """
    mix = mix.copy()
    dead = (mix["n_mass"] if "n_mass" in mix else mix["n_live"]) == 0
    mix["tier"] = np.where(
        dead, "unusable",
        np.where(mix["purity"] >= core_purity, "core",
                 np.where(mix["purity"] <= foreign_max, "foreign", "transition")))
    return mix


# ==========================================================================
# 4. read-outs
# ==========================================================================
def dependence_matrix(res: dict, kind: str = "flow", top: int = 5,
                      use: str = "live") -> pd.DataFrame:
    """District x district influence, read off each district's best gauges.

    Rows are the OBSERVING district, columns the district whose drift the
    observation contains. Asymmetric on purpose -- influence follows the
    supply path.

    `use` DEFAULTS TO "live" AND THIS MATTERS. Built on `core` gauges the
    matrix comes out near-identity (1.000 on the diagonal, 0.000 off it) on
    every network, which is not a finding: a core gauge is DEFINED as one with
    purity >= `core_purity`, so restricting to cores and then measuring purity
    is circular. The informative matrix is over every gauge that cleared the
    null, whatever its purity -- that is the population a real placement draws
    from. `use="core"` is kept as a sanity check: it should be ~identity, and
    if it is not, the tiering is inconsistent with the weights.

    This is a dependence GROUND TRUTH in the sense the project has been
    missing: measured from real drifts at the real operating point, not
    inferred from the observables the FL side will also be trained on.
    Influence is not causality in the time-series sense, and nothing here
    licenses a directional-causality claim; it licenses a statement about
    whose demand change reaches whose meters.
    """
    m = res["mixture"]
    m = m[m["kind"] == kind]
    if use == "core":
        m = m[m["tier"] == "core"]
    elif use == "live":
        m = m[m["tier"] != "unusable"]
    elif use != "all":
        raise KeyError("use must be live | core | all")
    m = m[~m.get("degenerate_null", False).astype(bool)] \
        if "degenerate_null" in m else m
    ds = res["districts"]
    wcols = [f"w_{d}" for d in ds]
    rows = {}
    for d in ds:
        g = m[m["home"] == d].nlargest(top, "snr_home") if top else m[m["home"] == d]
        rows[d] = (g[wcols].mean().to_numpy() if len(g)
                   else np.full(len(ds), np.nan))
    return pd.DataFrame({d: rows[d] for d in ds}, index=ds).T


def dependence_spread(res: dict, kind: str = "flow", top: int = 5) -> pd.DataFrame:
    """The same matrix under three gauge populations, side by side.

    `core` should be ~identity by construction; if `live` is also ~identity
    the network really is separable, and if it is not, the difference between
    the two is exactly what tiering throws away.
    """
    rows = []
    for use in ("core", "live", "all"):
        A = dependence_matrix(res, kind=kind, top=top, use=use).to_numpy()
        off = A[~np.eye(len(A), dtype=bool)]
        rows.append({"use": use, "kind": kind,
                     "diag": float(np.nanmean(np.diag(A))),
                     "offdiag": float(np.nanmean(off)),
                     "offdiag_max": float(np.nanmax(off)),
                     "ratio": float(np.nanmean(np.diag(A)) /
                                    max(float(np.nanmean(off)), 1e-9))})
    return pd.DataFrame(rows).round(3)


def core_connectivity(P: sp.Probe, res: dict, kind: str = "flow") -> pd.DataFrame:
    """Is a district's core set contiguous on the network graph?

    Measured, not assumed. Hydraulic influence follows the supply path rather
    than Euclidean distance, so a core set has no a-priori reason to be a
    connected patch. If it is, a placement rule can be stated geometrically;
    if it is not, placement has to name individual elements.
    """
    import networkx as nx
    G = nx.Graph()
    ends = sp._link_endpoints(P)
    for name, (u, v, _) in ends.items():
        G.add_edge(u, v, link=name)

    m = res["mixture"]
    rows = []
    for d in res["districts"]:
        core = m[(m["home"] == d) & (m["kind"] == kind) & (m["tier"] == "core")]
        if not len(core):
            rows.append({"district": d, "kind": kind, "n_core": 0,
                         "components": 0, "largest_share": np.nan})
            continue
        if kind == "pressure":
            nodes = set(core["element"])
        else:
            nodes = set()
            for e in core["element"]:
                u, v, _ = ends[e]
                nodes |= {u, v}
        sub = G.subgraph(nodes)
        comps = list(nx.connected_components(sub))
        rows.append({"district": d, "kind": kind, "n_core": len(core),
                     "components": len(comps),
                     "largest_share": (max(len(c) for c in comps) / len(nodes)
                                       if nodes else np.nan)})
    return pd.DataFrame(rows)


def beta_stability(res_a: dict, res_b: dict, label_a="beta_a",
                   label_b="beta_b") -> pd.DataFrame:
    """Do the mixture weights survive a change of drift strength?

    Invariant weights mean the network is operating quasi-linearly and the
    mixture is a property of the topology, transferable across scenarios.
    Weights that move mean it is an operating-point statement and has to be
    re-measured -- which is the open question the ky17 Spearman drop left.
    """
    ds = res_a["districts"]
    w = [f"w_{d}" for d in ds]
    a = res_a["mixture"].set_index("id")
    b = res_b["mixture"].set_index("id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    A, B = a[w].to_numpy(), b[w].to_numpy()
    l1 = 0.5 * np.abs(A - B).sum(axis=1)              # total variation on a simplex
    rows = []
    for kind in sorted(set(a["kind"])):
        m = (a["kind"] == kind).to_numpy()
        rows.append({
            "kind": kind, "n": int(m.sum()),
            "purity_corr": float(np.corrcoef(a.loc[m, "purity"].fillna(0),
                                             b.loc[m, "purity"].fillna(0))[0, 1]),
            "tv_mean": float(np.nanmean(l1[m])), "tv_p90": float(np.nanquantile(l1[m], .9)),
            "tier_agree": float((a.loc[m, "tier"] == b.loc[m, "tier"]).mean()),
        })
    return pd.DataFrame(rows).assign(compare=f"{label_a} vs {label_b}").round(3)


def zones(res: dict, kind: str = "flow", core_purity: float = 0.85,
          band: tuple = (0.4, 0.85)) -> pd.DataFrame:
    """Placement zones, per district: where a gauge can be put and read.

    Not an optimiser. It says which elements are in a REGION whose content is
    known -- pure, quantifiably mixed, or unreadable -- so a placement can
    avoid the last category without being tuned against a downstream score.
    """
    m = res["mixture"]
    m = m[m["kind"] == kind]
    rows = []
    for d, g in m.groupby("home"):
        live = g[g["tier"] != "unusable"]
        rows.append({
            "district": d, "kind": kind, "n": len(g),
            "core": int((g["tier"] == "core").sum()),
            "transition": int((g["tier"] == "transition").sum()),
            "foreign": int((g["tier"] == "foreign").sum()),
            "unusable": int((g["tier"] == "unusable").sum()),
            "readable_band": int(live["purity"].between(*band).sum()),
            "purity_max": float(g["purity"].max()),
            "purity_median": float(g["purity"].median()),
            "top_second": (live["second"].mode().iat[0]
                           if len(live) and len(live["second"].mode()) else ""),
        })
    return pd.DataFrame(rows).round(3)



# ==========================================================================
# 4b. DEPENDENCE -- the corrected estimator
# ==========================================================================
# The share-based `dependence_matrix` above answers "of what this gauge saw,
# what fraction came from district k". That is a composition, not a
# dependence, and it has four defects that this section fixes.
#
# 1. IT DISCARDS MAGNITUDE. Two gauges whose absolute responses differ by
#    1000x give identical rows once normalised to a simplex. "How much does
#    j depend on k" is a gain, not a share.
# 2. THE PER-WORLD NORMALISATION BIASES IT. g = <dz_s, u_k>/||u_k||^2 divides
#    by the size of THAT world's excitation, so a district whose conversion
#    produced a larger shape change is systematically under-credited. Shares
#    built from those g's are not comparable across k.
# 3. THE HARD `live` GATE MANUFACTURES SEPARABILITY. Gating on ||dz|| before
#    normalising makes rows one-hot as soon as only one district clears, which
#    is exactly what produced diag 1.000 / offdiag 0.000 on KY7 and D-Town at
#    null_factor 3, against 0.69/0.077 on Graeme at 1.5. The threshold was
#    reporting its own setting as a property of the network.
# 4. THE ROWS ARE NOT COMPARABLE. Nothing normalises a district's response to
#    its own scale, so a well-metered district and a poorly-metered one are
#    read on different rulers.
#
# The corrected chain:
#
#     u_k    = dz_d(k) / ||dz_d(k)||           unit direction of k's change
#     P[s,k] = <dz_s^(k), u_k>                 response along it
#     nu     = |<split-half dz_s, u_k>|        noise ON THE SAME DIRECTION
#     P~     = sign(P) * max(|P| - nu, 0)      SOFT threshold, no cliff
#     E[s,k] = P~[s,k] / ||dz_d(k)||           per unit of excitation
#
#     D[j,k] = mean over district j's gauges of |E[s,k]|        (absolute)
#     R[j,k] = D[j,k] / D[j,j]                                  (relative)
#
# `R` is the object the thesis needs: "a drift in k moves district j's meters
# R[j,k] as much as an equivalent drift in j does". Its diagonal is 1 by
# construction, the same gauges appear in numerator and denominator so the
# placement confound largely cancels, and it is asymmetric, which is the
# point.
#
# NAMING, because the over-claim is easy to make: this is OBSERVATIONAL
# dependence. Each district's demand is generated independently in the
# simulator, so nothing here says client j's demand depends on client k's. It
# says client j's OBSERVATIONS are contaminated by client k's demand -- which
# is precisely the confound federated training faces, and the reason a
# measured version of it is worth having.


def elasticity(base: dict, shrink: bool = True,
               space: str = "shape") -> pd.DataFrame:
    """E[s,k]: sensor s's response per unit of district k's excitation.

    `space="shape"` works on z-scored weekly profiles: dimensionless, and the
    part of the response that survives the per-client scaler, so it is the
    FL-relevant form. `space="native"` works in mca and L/s: the linearised
    operator itself, dimensioned, and the form that should be invariant to the
    size of the drift. Report both -- where they disagree, the disagreement is
    the finding.

    Soft-thresholded against the projection noise, so a gauge that did not
    respond decays to 0 smoothly instead of being cut at a threshold.
    """
    sfx = "" if space == "shape" else "_native"
    if f"PROJ{sfx}" not in base:
        raise KeyError(f"space={space!r} needs a mixture built with the "
                       "current mixture_probe; rebuild it")
    Pj, Nu = base[f"PROJ{sfx}"].to_numpy(), base[f"NU{sfx}"].to_numpy()
    exc = base[f"EXC{sfx}"].to_numpy()[None, :]
    Pt = np.sign(Pj) * np.maximum(np.abs(Pj) - Nu, 0.0) if shrink else Pj
    return pd.DataFrame(Pt / np.where(exc > EPS, exc, np.nan),
                        index=base[f"PROJ{sfx}"].index,
                        columns=base[f"PROJ{sfx}"].columns)


def dependence(base: dict, kind: str = "flow", top: int | None = None,
               shrink: bool = True, stat: str = "mean",
               tier: str | None = None, mixture: pd.DataFrame | None = None,
               space: str = "shape"):
    """Observational dependence between districts. Returns (D, R, meta).

    `D[j,k]` is district j's mean |elasticity| to district k's drift, in
    z-units of the sensor per unit of demand-shape change. `R = D / diag(D)`
    is the relative form and is the one to quote.

    `top=None` uses EVERY gauge of that kind homed in j -- deliberately. Taking
    the best few by own-district SNR selects the purest gauges and drives the
    off-diagonal to zero by construction; that circularity is what the share
    estimator suffered from. Pass `top` only to ask a placement question
    ("what would the best k gauges see"), never to characterise the network.
    """
    E = elasticity(base, shrink=shrink, space=space)
    cand = base["cand"].set_index("id")
    ds = base["districts"]
    home = cand["district"].reindex(E.index)
    kinds = cand["kind"].reindex(E.index)
    keep = kinds == kind
    if tier is not None:
        if mixture is None:
            raise ValueError("tier filtering needs the mixture frame")
        t = mixture.set_index("id")["tier"].reindex(E.index)
        keep &= (t == tier)
    # The degenerate flag lives in `base["degenerate"]`, NOT in `cand` -- the
    # candidate frame comes from `sp.candidates` and has never carried that
    # column, so the guard that read it here never fired. `dependence_matrix`
    # reads the same flag off the mixture frame and DOES exclude them, so the
    # two estimators were running on different populations.
    #
    # The bias had a direction. A degenerate gauge is one whose init series is
    # constant (a closed valve, a pump on fixed duty): dz ~ 0, so E ~ 0. Those
    # zeros deflate D[j,j] and D[j,k] alike, but the diagonal is the larger
    # quantity and loses proportionally more -- so R = D/diag(D) came out too
    # coupled, with off-diagonals inflated by gauges that never moved.
    deg = base.get("degenerate")
    if deg is not None:
        keep &= ~pd.Series(deg).reindex(E.index).fillna(False).astype(bool)

    rows, n = {}, {}
    for j in ds:
        sel = E[keep & (home == j)]
        if top and len(sel):
            snr = (base["DZ"].loc[sel.index, j] / base["null"].loc[sel.index])
            sel = sel.loc[snr.nlargest(top).index]
        n[j] = len(sel)
        v = sel.abs()
        rows[j] = (getattr(v, stat)().to_numpy() if len(sel)
                   else np.full(len(ds), np.nan))
    D = pd.DataFrame({j: rows[j] for j in ds}, index=ds).T
    diag = pd.Series(np.diag(D.to_numpy()), index=ds)
    R = D.div(diag.replace(0, np.nan), axis=0)
    meta = pd.Series(n, name="n_gauges")
    return D, R, meta


def dependence_summary(R: pd.DataFrame) -> pd.DataFrame:
    """Per district: how much it leaks out, how much leaks in, net direction.

    `in_` is the mean of row j off-diagonal -- how contaminated j's meters are.
    `out` is the mean of column j -- how much j's drift shows up elsewhere. A
    district with high `out` and low `in_` is an upstream driver; the reverse
    is a downstream sink. `net` is the difference, and it is the closest thing
    to a direction this design supports.
    """
    A = R.to_numpy().copy()
    np.fill_diagonal(A, np.nan)
    out = pd.DataFrame({
        "in_mean": np.nanmean(A, axis=1), "in_max": np.nanmax(A, axis=1),
        "out_mean": np.nanmean(A, axis=0), "out_max": np.nanmax(A, axis=0),
    }, index=R.index)
    out["net"] = out["out_mean"] - out["in_mean"]
    out["strongest_source"] = [R.columns[i] for i in np.nanargmax(A, axis=1)]
    return out.round(3)


def asymmetry(R: pd.DataFrame) -> pd.DataFrame:
    """R[j,k] - R[k,j]. Positive means k influences j more than j influences k."""
    return (R - R.T).round(3)


def estimator_checks(base: dict, kind: str = "flow",
                     space: str = "shape") -> pd.DataFrame:
    """Does the estimator behave the way its derivation says it should?

    `noise_pass_rate` -- share of (gauge, district) cells whose projection is
    below its own noise floor. With D districts and one real driver per gauge
    this should be substantial; near 0 means the noise floor is under-
    estimated and everything looks significant.

    `shrink_effect` -- correlation between raw and shrunk elasticities. Near 1
    means soft-thresholding changed rankings very little and the result does
    not hinge on it.

    `diag_dominance` -- share of gauges whose largest |E| is their own
    district. This is the honest version of "hit rate": it uses no threshold
    and no tiering.
    """
    sfx = "" if space == "shape" else "_native"
    Pj, Nu = base[f"PROJ{sfx}"].to_numpy(), base[f"NU{sfx}"].to_numpy()
    cand = base["cand"].set_index("id")
    keep = (cand["kind"].reindex(base[f"PROJ{sfx}"].index) == kind).to_numpy()
    ds = base["districts"]
    home = cand["district"].reindex(base[f"PROJ{sfx}"].index).to_numpy()
    hidx = np.array([ds.index(h) if h in ds else -1 for h in home])

    Er = elasticity(base, shrink=False, space=space).to_numpy()[keep]
    Es = elasticity(base, shrink=True, space=space).to_numpy()[keep]
    ok = np.isfinite(Er) & np.isfinite(Es)
    big = np.argmax(np.abs(Es), axis=1)
    return pd.DataFrame([{
        "kind": kind, "space": space, "n_gauges": int(keep.sum()),
        "noise_pass_rate": float((np.abs(Pj[keep]) <= Nu[keep]).mean()),
        "shrink_effect_r": float(np.corrcoef(Er[ok].ravel(), Es[ok].ravel())[0, 1]),
        "diag_dominance": float((big == hidx[keep]).mean()),
        "median_nu_over_proj": float(np.median(
            Nu[keep] / np.maximum(np.abs(Pj[keep]), EPS))),
    }]).round(4)


def beta_linearity(base_a: dict, base_b: dict, kind: str = "flow",
                   label_a: str = "a", label_b: str = "b",
                   space: str = "shape") -> pd.DataFrame:
    """Is the response operator linear in the excitation?

    `E` is already per unit of excitation, so if the network responds linearly
    E is beta-invariant: correlation ~1 and slope ~1. A slope away from 1 is
    the size of the non-linearity, and it is the quantity that decides whether
    a dependence matrix measured at one operating point can be quoted at
    another.
    """
    Ea, Eb = elasticity(base_a, space=space), elasticity(base_b, space=space)
    idx = Ea.index.intersection(Eb.index)
    cand = base_a["cand"].set_index("id")
    idx = idx[(cand["kind"].reindex(idx) == kind).to_numpy()]
    x = Ea.loc[idx].to_numpy().ravel()
    y = Eb.loc[idx].to_numpy().ravel()
    ok = np.isfinite(x) & np.isfinite(y) & ((np.abs(x) + np.abs(y)) > EPS)
    x, y = x[ok], y[ok]
    slope = float(x @ y / (x @ x)) if len(x) and (x @ x) > EPS else np.nan
    return pd.DataFrame([{
        "kind": kind, "space": space,
        "compare": f"{label_a} vs {label_b}", "n": int(len(x)),
        "r": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan,
        "slope": slope,
        "median_ratio": float(np.median(y[np.abs(x) > EPS] / x[np.abs(x) > EPS]))
        if (np.abs(x) > EPS).any() else np.nan,
    }]).round(3)


def response_matrix(base: dict, kind: str = "flow", top: int | None = 3,
                    tier: str | None = None, mixture: pd.DataFrame | None = None,
                    normalise: bool = True, space: str = "shape") -> pd.DataFrame:
    """Rows = the district that DRIFTED, columns = where it was measured.

    The transpose-orientation counterpart of `dependence`, matching the sweep's
    `leak_matrix` layout so the two can be read together. With
    `normalise=True` each row is divided by its own diagonal, so an entry is
    "this drift showed up at j's meters x as strongly as at its own".
    """
    D, _, _ = dependence(base, kind=kind, top=top, tier=tier, mixture=mixture,
                         space=space)
    M = D.T                                   # rows: drifted, cols: observed
    if normalise:
        diag = pd.Series(np.diag(M.to_numpy()), index=M.index)
        M = M.div(diag.replace(0, np.nan), axis=0)
    return M



# ==========================================================================
# 4c. why is one network noisier than another?
# ==========================================================================
def null_vs_storage(P: sp.Probe, base: dict, kind: str = "pressure") -> dict:
    """Regress each gauge's noise floor on its distance to storage.

    The motivating observation: the baseline split-half null is ~0.27 on
    Graeme, ~1.5 on D-Town and ~4.7 on KY7 -- and KY7 is the network whose
    three tanks float the whole system above the reservoir head. If a gauge's
    own noise floor grows with proximity to storage, then "how much storage
    floats this district" is a pre-flight predictor of whether a network can
    support this analysis at all, computable before any simulation.

    Model: log(null) ~ hops_to_storage + elevation (pressure) or
    log|mean flow| (flow). Hops are unweighted shortest-path steps on the pipe
    graph to the nearest tank or reservoir -- crude, but it needs no
    calibration and no assumption about which tank serves what.

    Interpretation guard: a negative `hops` coefficient means noise FALLS with
    distance from storage, i.e. storage is the noise source. A flat fit does
    not clear storage -- it only says this predictor does not carry it.
    """
    import networkx as nx

    ends = sp._link_endpoints(P)
    G = nx.Graph()
    for name, (u, v, _) in ends.items():
        G.add_edge(u, v)
    storage = set()
    for attr in ("tank_name_list", "reservoir_name_list"):
        storage |= set(getattr(P.wn, attr, []) or [])
    storage &= set(G.nodes)
    if not storage:
        return {"table": pd.DataFrame(), "fit": None,
                "note": "no tank or reservoir on the pipe graph"}

    # one multi-source BFS instead of one per gauge
    G.add_node("__STORAGE__")
    for t in storage:
        G.add_edge("__STORAGE__", t)
    hops = nx.single_source_shortest_path_length(G, "__STORAGE__")
    hops = {n: h - 1 for n, h in hops.items() if n != "__STORAGE__"}

    null = base["null"]
    cand = base["cand"].set_index("id")
    rows = []
    mean_q = P.flows.drop(columns=["month"], errors="ignore").abs().mean()
    for gid, r in cand.iterrows():
        if r["kind"] != kind or gid not in null.index:
            continue
        if kind == "pressure":
            h = hops.get(r["element"])
            extra = float(getattr(P.wn.get_node(r["element"]), "elevation", np.nan))
            xname = "elevation"
        else:
            u, v, _ = ends.get(r["element"], (None, None, None))
            hs = [hops.get(x) for x in (u, v) if hops.get(x) is not None]
            h = min(hs) if hs else None
            extra = float(np.log10(max(float(mean_q.get(r["element"], 0.0)), 1e-6)))
            xname = "log10_mean_flow"
        if h is None or not np.isfinite(extra):
            continue
        rows.append({"id": gid, "district": r["district"], "element": r["element"],
                     "hops_to_storage": int(h), xname: extra,
                     "null": float(null.loc[gid])})
    tab = pd.DataFrame(rows)
    if len(tab) < 10:
        return {"table": tab, "fit": None, "note": "too few gauges to fit"}

    y = np.log10(np.maximum(tab["null"].to_numpy(), 1e-9))
    X = np.column_stack([np.ones(len(tab)), tab["hops_to_storage"].to_numpy(),
                         tab[xname].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    fit = pd.DataFrame([{
        "kind": kind, "n": len(tab), "r2": 1 - ss_res / max(ss_tot, EPS),
        "intercept": coef[0], "b_hops": coef[1], f"b_{xname}": coef[2],
        "r_hops_only": float(np.corrcoef(tab["hops_to_storage"], y)[0, 1]),
        "null_median": float(tab["null"].median()),
        "hops_median": float(tab["hops_to_storage"].median()),
    }]).round(4)
    return {"table": tab, "fit": fit, "xname": xname, "note": ""}


def plot_null_vs_storage(res_ns: dict, ax=None, label: str = ""):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))
    t = res_ns["table"]
    if not len(t):
        return ax
    ax.scatter(t["hops_to_storage"], t["null"], s=12, alpha=.5)
    med = t.groupby("hops_to_storage")["null"].median()
    ax.plot(med.index, med.to_numpy(), color="crimson", lw=1.6, marker="o", ms=3)
    ax.set(yscale="log", xlabel="hops to nearest tank / reservoir",
           ylabel="split-half null", title=label)
    return ax


# ==========================================================================
# 5. figures
# ==========================================================================
def plot_settle(P: sp.Probe, ax=None, ref_months: int = 3,
                slack: float = 0.01):
    """Correlation to the terminal profile, month by month."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.4))
    scan = settle_scan(P, ref_months)
    tgt = P.drift.get("tgt_district")
    for d, g in scan.groupby("district"):
        hot = d == tgt
        ax.plot(g["month"], g["r_to_terminal"], marker="o", ms=3, label=d,
                lw=2.2 if hot else 1.0, alpha=1.0 if hot else 0.4)
    rt = scan[scan["district"] == tgt].sort_values("month")["r_to_terminal"].to_numpy()
    ax.axhline(_threshold(rt, ref_months, slack), color="crimson", ls="--", lw=.8)
    if len(P.schedule):
        ax.axvline(int(P.schedule["drift_month"].max()), color="k", ls=":", lw=.9)
    s = settle_month(P, ref_months, slack).get(tgt)
    if s is not None:
        ax.axvline(s, color="tab:green", lw=1.2)
    ax.set(xlabel="month", ylabel="r to terminal weekly profile",
           title="settling — last switch (dotted) to settled (green)")
    ax.legend(fontsize=6, ncol=2)
    return ax


def plot_purity(res: dict, kind: str = "flow", ax=None):
    """Purity against response strength. Tiers fall out as regions."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.2))
    m = res["mixture"]
    m = m[m["kind"] == kind]
    for tier, c in (("core", "tab:green"), ("transition", "tab:orange"),
                    ("foreign", "tab:red"), ("unusable", "lightgrey")):
        g = m[m["tier"] == tier]
        ax.scatter(g["snr_home"].clip(upper=60), g["purity"], s=16, alpha=.65,
                   c=c, label=f"{tier} ({len(g)})")
    ax.axhline(0.85, color="k", lw=.6, ls="--")
    ax.set(xlabel="response / split-half null  (own district)", ylabel="purity",
           title=f"{kind}: mixture purity vs response strength")
    ax.legend(fontsize=7)
    return ax


def plot_dependence(A: pd.DataFrame, ax=None, title: str = ""):
    """The district x district influence matrix."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.2))
    V = A.to_numpy()
    ax.imshow(V, cmap="magma", vmin=0, vmax=1)
    ax.set(xticks=range(A.shape[1]), yticks=range(A.shape[0]),
           xlabel="whose drift is in the signal", ylabel="observing district",
           title=title or "influence mixture")
    ax.set_xticklabels([c.replace("District_", "") for c in A.columns])
    ax.set_yticklabels([c.replace("District_", "") for c in A.index])
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if np.isfinite(V[i, j]):
                ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="w" if V[i, j] < 0.6 else "k")
    return ax


def plot_matrix(A: pd.DataFrame, ax=None, title: str = "", vmax: float | None = None,
                xlabel: str = "", ylabel: str = "", cmap: str = "magma"):
    """Annotated heatmap for any district x district matrix."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.2))
    V = A.to_numpy(dtype=float)
    hi = vmax if vmax is not None else float(np.nanmax(V)) or 1.0
    ax.imshow(V, cmap=cmap, vmin=0, vmax=hi)
    ax.set(xticks=range(A.shape[1]), yticks=range(A.shape[0]),
           xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_xticklabels([str(c).replace("District_", "") for c in A.columns])
    ax.set_yticklabels([str(c).replace("District_", "") for c in A.index])
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if np.isfinite(V[i, j]):
                ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="w" if V[i, j] < 0.6 * hi else "k")
    return ax


def plot_mixture_bars(res: dict, district: str, kind: str = "flow",
                      top: int = 12, ax=None):
    """Stacked simplex per gauge, ordered by purity. The literal read-out."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.4))
    ds = res["districts"]
    m = res["mixture"]
    g = (m[(m["home"] == district) & (m["kind"] == kind)
           & (m["tier"] != "unusable")]
         .nlargest(top, "snr_home").sort_values("purity", ascending=False))
    bottom = np.zeros(len(g))
    for d in ds:
        v = g[f"w_{d}"].to_numpy()
        ax.bar(np.arange(len(g)), v, bottom=bottom, label=d.replace("District_", ""))
        bottom += v
    ax.set_xticks(np.arange(len(g)))
    ax.set_xticklabels(g["id"], rotation=75, fontsize=6)
    ax.set(ylabel="mixture weight", ylim=(0, 1),
           title=f"{district} — {kind} gauges, what each is reading")
    ax.legend(fontsize=6, ncol=len(ds))
    return ax


def plot_tier_map(P: sp.Probe, res: dict, kind: str = "flow", ax=None):
    """The network drawn with each element coloured by tier.

    This is where "are the cores contiguous" is answered by eye; the number is
    in `core_connectivity`.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 6.4))
    pos = {n: P.wn.get_node(n).coordinates for n in P.wn.node_name_list}
    ends = sp._link_endpoints(P)
    G = nx.Graph()
    for name, (u, v, _) in ends.items():
        G.add_edge(u, v)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.88", width=0.7)

    m = res["mixture"]
    m = m[m["kind"] == kind].set_index("element")
    colour = {"core": "tab:green", "transition": "tab:orange",
              "foreign": "tab:red", "unusable": "lightgrey"}
    if kind == "pressure":
        for tier, c in colour.items():
            sel = [e for e in m.index[m["tier"] == tier] if e in pos]
            if sel:
                xy = np.array([pos[e] for e in sel])
                ax.scatter(xy[:, 0], xy[:, 1], s=14, c=c, label=tier, zorder=3)
    else:
        for tier, c in colour.items():
            for e in m.index[m["tier"] == tier]:
                if e not in ends:
                    continue
                u, v, _ = ends[e]
                if u in pos and v in pos:
                    ax.plot(*zip(pos[u], pos[v]), color=c, lw=1.8, zorder=3,
                            solid_capstyle="round")
        for tier, c in colour.items():
            ax.plot([], [], color=c, lw=2, label=tier)
    ax.set(xticks=[], yticks=[], title=f"{kind} tiers")
    ax.legend(fontsize=7)
    return ax
