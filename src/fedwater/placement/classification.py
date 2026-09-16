"""classification.py -- tiers, blend shape, channel diversity, report-only arms.

(Ported from the POC's ``placement.py``. The shipped placement is now the
slot composite in :mod:`fedwater.placement.select`; the four arms below are
kept as REPORT-ONLY classifications -- ``control_low`` and ``unusable`` in
particular -- and their coverage is written beside the placement.)

Original notes follow.

What this produces
------------------
Per configuration, four complete `sensors:` blocks, each of shape
`{district: {"pressure": [...], "flow": [...]}}` with at most `n_max` of each
kind per district. Together with `districts.yml` they are the
`(district attribution, sensor placement)` tuple the federated step consumes,
and having several of them means the FL side can run matched arms rather than
one placement and a shrug.

    control_high   gauges that read almost only their own district.
    control_low    gauges that respond strongly and to the WRONG district --
                   transit pipes whose flow reverses when the local district
                   draws more, so a client metered here is reading a
                   neighbour. The informative failure.
    target         gauges carrying a QUANTIFIED BLEND of two NAMED districts.
                   The set the thesis needs.
    unusable       gauges whose signature did not move for any drift. Report
                   only: a client metered here gets a flat series and the FL
                   run degenerates rather than failing informatively.

Three things this module learned from the D-Town and KY7 stores
---------------------------------------------------------------
**1. The target cannot be a purity band.** It was `tier == "transition"` with
purity in [0.25, 0.75]. On D-Town the dependence does sit in `transition`; on
KY7 it sits in `foreign`, where a link inside district j is dominated by k's
transit. A band anchored on the HOME share does not transfer between the two.
What does transfer is TOP-TWO CONCENTRATION: `w1 + w2 >= 0.90` with
`w2 >= 0.20` admits a 60/40 and a 20/80 alike and rejects a smear across five
districts, whichever district happens to be the home one.

**2. A mixture is only informative if it VARIES BETWEEN GAUGES.** This is the
one that matters. On KY7 every pressure gauge reports the same simplex row --
the interquartile range of the largest weight spans 0.005 across 463 gauges,
because pressure there is the tank-and-pump state and has a single degree of
freedom. Per gauge it looks like a textbook 49/44 two-district blend and sails
through the top-two gate; 456 of them passed, containing 28 distinct readings.

The gauge-level test does not catch this. A uniform-smear detector
(`hhi ~ 1/K` with all districts live) flags 97% of KY7 fast_greedy pressure
and 0.0% of KY7 girvan_newman pressure -- which is the MORE degenerate channel
by every diversity measure -- while falsely flagging 19-28% of D-Town pressure,
which is genuinely diverse. Degeneracy is a property of the CHANNEL, not of any
one gauge's simplex row, so `channel_diversity` measures it across gauges:
mean pairwise total variation comes out 0.048-0.104 on KY7 pressure against
0.63-0.78 on every other channel in the store, a clean six-fold separation.

**3. Redundancy has to be broken inside the set.** Even on a healthy channel,
adjacent links in series carry identical flow: D-Town's District_A target set
was `q_P237`/`q_P379` at 0.433/3.971 and three more at 0.426/3.950. Five slots,
one measurement. `dedupe` enforces a minimum mixture distance.

Ragged sets are shipped, not padded
-----------------------------------
A district can simply have no gauge meeting the criteria. Padding to `n_max`
from outside them would put non-targets in the target arm, which is the one
thing this module must not do. So short sets ship and every row carries
`n_selected` against `n_eligible`. The cost is named here rather than
discovered later: client CSVs then differ in column count per kind, and
`preprocess_clients` reads that as a different feature geometry per client.

A note on `r_res_own_init`
--------------------------
Reported, never ranked on, and not to be read as a quality signal. The world
family starts from an all-residential map, so at `init` every district carries
the same regime token and there is nothing for a resemblance statistic to
resolve: mean `r_res_own` is 0.117 (D-Town flow) and 0.097 (KY7 flow) at init
against 0.486 and 0.414 at final, and `r_res_other_regime` comes back NaN
because no peer district differs. A target gauge with `r_res_own_init ~ 0` is
the experiment design showing through, not a bad gauge.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from . import signal_probe as sp

__all__ = ["ARMS", "PIPELINE_ARMS", "ArmSpec", "classification",
           "mixture_matrix", "channel_diversity", "eligible", "dedupe",
           "select", "selection_table", "coverage", "sensors_yaml",
           "rebuild_mixture", "CHANNEL_MIN_TV", "CHANNEL_MIN_DISTINCT",
           "DEDUPE_MIN_DISTANCE"]

# Channel verdict thresholds. The observed gap is 0.048-0.104 (degenerate)
# against 0.628-0.793 (healthy), so 0.25 sits in open space with room on both
# sides rather than being tuned to either.
CHANNEL_MIN_TV = 0.25
CHANNEL_MIN_DISTINCT = 0.12

# Two gauges whose simplex rows differ by less than this are the same reading.
# Total variation, so 0.10 means "ten percent of the mass sits differently".
DEDUPE_MIN_DISTANCE = 0.10


# ==========================================================================
# arm definitions
# ==========================================================================
class ArmSpec:
    """One arm: which gauges qualify, and in what order they are preferred."""

    def __init__(self, name, tier=None, min_top2=0.0, min_w2=0.0,
                 max_w2=1.0, min_n_mass=1, require_stable=False,
                 max_purity_swing=None, order=(), ascending=(),
                 dedupe_on="mixture", doc=""):
        self.name = name
        self.dedupe_on = dedupe_on
        self.tier = tier
        self.min_top2 = float(min_top2)
        self.min_w2 = float(min_w2)
        self.max_w2 = float(max_w2)
        self.min_n_mass = int(min_n_mass)
        self.require_stable = bool(require_stable)
        self.max_purity_swing = max_purity_swing
        self.order = tuple(order)
        self.ascending = tuple(ascending)
        self.doc = doc


ARMS = {
    "control_high": ArmSpec(
        "control_high", tier="core", min_top2=0.90, max_w2=0.15,
        order=("purity", "snr_home"), ascending=(False, False),
        dedupe_on="profile",
        doc="reads almost only its own district"),
    "control_low": ArmSpec(
        # Top-two concentration here too: a foreign gauge reading ONE wrong
        # district is a clean control; one smeared over three is noise wearing
        # a label. Deliberately no `min_n_mass` -- a single clean foreign
        # driver is the best version of this arm, not a disqualification.
        "control_low", tier="foreign", min_top2=0.90,
        order=("purity", "snr_home"), ascending=(True, False),
        dedupe_on="profile",
        doc="responds strongly, and to another district"),
    "target": ArmSpec(
        # No tier constraint and no home-share band: see note 1 above. The
        # blend is defined on the top two weights, whichever districts they
        # belong to.
        # `require_stable` is deliberately OFF. It tests whether a gauge keeps
        # the same TIER LABEL across null factors, which was the right test
        # when the arm was tier-defined and is a leftover now that it is not --
        # a gauge sitting near the core/transition boundary flips its label
        # while its weights barely move. `max_purity_swing` is the test that
        # survives: it asks whether the BLEND holds, which is what is being
        # claimed. Measured on the store, the difference is the whole result:
        # with the tier gate, D-Town girvan_newman covers 3 of 5 districts with
        # 9 gauges; without it, 5 of 5 with 14. Loosening the swing to 0.30
        # instead adds one gauge and changes no coverage, so 0.20 is not a
        # tuned number -- it is the flat part of the curve.
        "target", min_top2=0.90, min_w2=0.20, min_n_mass=2,
        require_stable=False, max_purity_swing=0.20,
        # Most stable blend first, strongest response as the tie-break. Not
        # "closest to 50/50": a 25/75 that holds across thresholds is a better
        # quantified mixture than a 50/50 that is an artefact of one.
        order=("purity_swing", "snr_home"), ascending=(True, False),
        dedupe_on="mixture",
        doc="a quantified blend of two named districts"),
    "unusable": ArmSpec(
        "unusable", tier="unusable",
        order=("snr_home",), ascending=(True,), dedupe_on="none",
        doc="did not move for any drift -- report only"),
}

PIPELINE_ARMS = ("control_high", "control_low", "target")


# ==========================================================================
# the classification table
# ==========================================================================
_NOT_DISTRICT_W = frozenset({"w_second"})


def _w_cols(table) -> list[str]:
    """The simplex columns, one per district (``w_<district>``).

    Was ``startswith("w_District")``, which silently assumed every partition
    names its districts ``District_*``. A bundle with ``DMA1``-style names
    would then have NO simplex columns, every mixture distance would be 0,
    and dedupe would collapse each district's mixed gauges into one.
    """
    return [c for c in table.columns
            if c.startswith("w_") and c not in _NOT_DISTRICT_W]


def classification(res) -> pd.DataFrame:
    """Every node and link of the configuration, tiered, with the blend shape.

    `w1`/`w2`/`top2` are the sorted simplex weights, and they are what the arms
    select on -- `purity` is still there, but it is the HOME share and so is
    partition-relative in a way the top two weights are not.
    """
    if not len(res.mixture):
        return pd.DataFrame()
    cols = ["id", "element", "kind", "home", "role", "tier", "purity",
            "second", "w_second", "snr_home", "n_live", "n_mass", "mass",
            "hhi", "entropy", "neg_mass", "degenerate_null"]
    m = res.mixture[[c for c in cols if c in res.mixture.columns]].copy()
    wcols = _w_cols(res.mixture)
    m = pd.concat([m, res.mixture[wcols]], axis=1)
    if len(res.stability):
        m = m.merge(res.stability, on="id", how="left")
    if len(res.scores):
        # The resemblance axis, carried alongside so an arm can be audited
        # against a score it was NOT chosen on. Read the module docstring
        # before treating the init column as a quality signal.
        s = (res.scores[res.scores["phase"] == "init"]
             .groupby("id")[["r_res_own", "response"]].mean().reset_index()
             .rename(columns={"r_res_own": "r_res_own_init",
                              "response": "response_mean"}))
        m = m.merge(s, on="id", how="left")
        if "served_own" in res.scores.columns:
            # The independent topological opinion: the share of downstream
            # demand a link carries for its own district. Correlates 0.40 with
            # purity on girvan_newman/D-Town, and links serving >=75% of their
            # own district's demand have median purity 1.000 -- so where
            # topology and hydraulics agree, the mixture agrees too.
            so = res.scores.groupby("id")["served_own"].mean().reset_index()
            m = m.merge(so, on="id", how="left")

    W = np.sort(m[wcols].fillna(0.0).to_numpy(float), axis=1)
    m["w1"], m["w2"] = W[:, -1], W[:, -2]
    m["top2"] = W[:, -1] + W[:, -2]
    return m.assign(config_id=res.config_id, method=res.method)


# ==========================================================================
# channel diversity -- does this channel carry more than one reading?
# ==========================================================================
def mixture_matrix(table: pd.DataFrame, kind: str):
    """(ids, simplex rows) for the live gauges of one channel."""
    wcols = _w_cols(table)
    d = table[(table["kind"] == kind) & (table["tier"] != "unusable")]
    return list(d["id"]), d[wcols].fillna(0.0).to_numpy(float)


def _mean_pairwise_tv(W: np.ndarray, cap: int = 400, seed: int = 0) -> float:
    """Mean total-variation distance between simplex rows, subsampled.

    The statistic that separates a channel carrying many readings from one
    carrying the same reading many times. Subsampled because it is O(n^2) and
    the estimate is stable well below 400 rows.

    Worth saying why the obvious cheaper statistics do NOT work here. The share
    of gauges on the modal simplex row does not separate: it runs 0.36-0.48 on
    KY7 flow (healthy) against 0.26-0.53 on KY7 pressure (degenerate), because
    the modal row of a healthy flow channel is the legitimate one-hot that
    every pure gauge shares. Pairwise distance is immune to that -- a channel
    with several one-hot clusters has distant pairs, a channel with one cluster
    does not.
    """
    if len(W) < 2:
        return 0.0
    if len(W) > cap:
        W = W[np.random.default_rng(seed).choice(len(W), cap, replace=False)]
    d = 0.5 * np.abs(W[:, None, :] - W[None, :, :]).sum(-1)
    return float(d[np.triu_indices(len(W), 1)].mean())


def channel_diversity(table: pd.DataFrame, min_tv: float = CHANNEL_MIN_TV,
                      min_distinct: float = CHANNEL_MIN_DISTINCT,
                      decimals: int = 2) -> pd.DataFrame:
    """Per kind: how many INDEPENDENT mixture readings the channel carries.

    `verdict` is `degenerate` when the gauges of a channel all report the same
    blend. Such a channel cannot support a placement decision at all -- every
    gauge is the same measurement, so `n_max` of them is one of them -- and
    `eligible` drops it from the arms. It is still classified and still
    reported, because "this channel has one degree of freedom" is a finding
    about the network, not a gap in the data.

    `eff_rank` is the participation ratio of the centred singular values: the
    number of independent directions the simplex rows span. It came out 1.12
    on KY7 girvan_newman pressure and 3.31-3.96 on every D-Town channel.
    """
    rows = []
    for kind in sorted(set(table["kind"])):
        ids, W = mixture_matrix(table, kind)
        if not len(W):
            rows.append({"kind": kind, "n_live": 0, "distinct_rows": 0,
                         "distinct_share": 0.0, "mean_pairwise_tv": 0.0,
                         "eff_rank": 0.0, "verdict": "empty"})
            continue
        Wc = W - W.mean(axis=0)
        s = np.linalg.svd(Wc, compute_uv=False)
        eff = float((s.sum() ** 2) / max((s ** 2).sum(), 1e-12))
        uniq = len(np.unique(np.round(W, decimals), axis=0))
        tv = _mean_pairwise_tv(W)
        ok = (tv >= min_tv) and (uniq / len(W) >= min_distinct)
        rows.append({"kind": kind, "n_live": len(W), "distinct_rows": uniq,
                     "distinct_share": round(uniq / len(W), 4),
                     "mean_pairwise_tv": round(tv, 4),
                     "eff_rank": round(eff, 3),
                     "verdict": "ok" if ok else "degenerate"})
    return pd.DataFrame(rows)


# ==========================================================================
# selection
# ==========================================================================
def eligible(table: pd.DataFrame, arm: str, channels: pd.DataFrame | None = None,
             enforce_channel: bool = True) -> pd.DataFrame:
    """Rows of `classification` that qualify for `arm`, unordered and uncapped."""
    spec = ARMS[arm]
    d = table
    if enforce_channel and arm != "unusable":
        ch = channel_diversity(table) if channels is None else channels
        bad = set(ch.loc[ch["verdict"] != "ok", "kind"])
        if bad:
            d = d[~d["kind"].isin(bad)]
    if spec.tier is not None:
        d = d[d["tier"] == spec.tier]
    if arm != "unusable":
        if spec.min_top2 > 0:
            d = d[d["top2"].fillna(0.0) >= spec.min_top2]
        if spec.min_w2 > 0:
            d = d[d["w2"].fillna(0.0) >= spec.min_w2]
        if spec.max_w2 < 1.0:
            d = d[d["w2"].fillna(0.0) <= spec.max_w2]
        if spec.min_n_mass > 1 and "n_mass" in d:
            d = d[d["n_mass"].fillna(0) >= spec.min_n_mass]
        if spec.require_stable and "tier_stable" in d:
            d = d[d["tier_stable"].fillna(False).astype(bool)]
        if spec.max_purity_swing is not None and "purity_swing" in d:
            d = d[d["purity_swing"].fillna(1.0) <= spec.max_purity_swing]
    if "degenerate_null" in d:
        # A gauge whose init series is constant has a null of ~0 and an SNR
        # that is not a measurement. It cannot be ranked on `snr_home`.
        d = d[~d["degenerate_null"].fillna(False).astype(bool)]
    return d


# What a control arm is deduplicated ON. Not the simplex row: a `core` gauge
# is one-hot BY DEFINITION and so is a `foreign` one, so every gauge of a
# district shares a row and mixture distance collapses the arm to one member
# per district. Measured on the real store, mixture-deduplicating
# `control_high` took 309 eligible D-Town flow gauges down to 8 -- the gate
# was destroying the arm, not cleaning it. These four columns separate gauges
# that a one-hot row cannot: two links in series carry identical flow and so
# agree on all four, while two genuinely different core gauges do not.
_PROFILE_COLS = ("snr_home", "purity", "served_own", "response_mean")


def _dedupe_space(d: pd.DataFrame, on: str):
    """The matrix distance is measured in, standardised where it is not already."""
    if on == "mixture":
        return d[_w_cols(d)].fillna(0.0).to_numpy(float), 0.5
    cols = [c for c in _PROFILE_COLS if c in d.columns]
    if not cols:
        return None, None
    V = d[cols].astype(float).to_numpy()
    V = np.where(np.isfinite(V), V, np.nan)
    mu = np.nanmean(V, axis=0)
    sd = np.nanstd(V, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = np.nan_to_num((V - mu) / sd)
    # mean absolute z-difference, so `min_distance` keeps the same meaning
    # ("this fraction of a standard deviation apart") across both spaces
    return Z, 1.0 / max(len(cols), 1)


def dedupe(d: pd.DataFrame, n_max: int,
           min_distance: float = DEDUPE_MIN_DISTANCE,
           on: str = "mixture") -> pd.DataFrame:
    """Walk `d` in its existing order, keeping rows that are new readings.

    Greedy rather than clustered, so the ranking still decides WHICH of a
    redundant group is kept -- the best-ranked one -- and the distance only
    decides that the rest are not additional information. Adjacent links in
    series carry identical flow and land at distance ~0 in either space.

    `on="mixture"` uses total variation between simplex rows, and is right
    exactly where the mixture IS the selection criterion -- the target arm.
    `on="profile"` uses the response profile instead, for arms whose simplex
    row is one-hot by construction. `on="none"` disables it.
    """
    if not len(d) or n_max <= 0:
        return d.head(0)
    if on == "none":
        return d.head(n_max)
    M, scale = _dedupe_space(d, on)
    if M is None:
        return d.head(n_max)
    keep, kept_rows = [], []
    for i in range(len(d)):
        if kept_rows:
            dist = scale * np.abs(np.array(kept_rows) - M[i]).sum(axis=1)
            if dist.min() < min_distance:
                continue
        keep.append(i)
        kept_rows.append(M[i])
        if len(keep) >= n_max:
            break
    return d.iloc[keep]


def _ranked(table, arm, district, kind, channels, enforce_channel):
    spec = ARMS[arm]
    d = eligible(table, arm, channels=channels, enforce_channel=enforce_channel)
    sub = d[(d["home"] == district) & (d["kind"] == kind)]
    order = [c for c in spec.order if c in sub.columns]
    asc = [a for c, a in zip(spec.order, spec.ascending) if c in sub.columns]
    return sub.sort_values(order, ascending=asc) if order else sub


def select(table: pd.DataFrame, districts, arm: str, n_max: int = 5,
           channels: pd.DataFrame | None = None, enforce_channel: bool = True,
           min_distance: float = DEDUPE_MIN_DISTANCE) -> dict:
    """The arm's `sensors:` block: <= `n_max` DISTINCT readings per district
    per kind."""
    channels = channel_diversity(table) if channels is None else channels
    out = {}
    for district in districts:
        pick = {}
        for kind in ("pressure", "flow"):
            sub = _ranked(table, arm, district, kind, channels, enforce_channel)
            sub = dedupe(sub, n_max, min_distance, on=ARMS[arm].dedupe_on)
            pick[kind] = [i[len(sp._PREFIX[kind]):] for i in sub["id"]]
        out[district] = pick
    return out


def selection_table(table: pd.DataFrame, districts, config_id,
                    arms=PIPELINE_ARMS, n_max: int = 5,
                    channels: pd.DataFrame | None = None,
                    enforce_channel: bool = True,
                    min_distance: float = DEDUPE_MIN_DISTANCE) -> pd.DataFrame:
    """Every selected gauge, with the values it was selected on.

    One row per (arm, district, kind, rank). The audit trail: an arm is only
    trustworthy if you can see the blend of each gauge it put in front of the
    trainer, and `n_eligible` against the row count shows how much of the
    eligible pool was redundant.
    """
    channels = channel_diversity(table) if channels is None else channels
    keep = ["id", "element", "kind", "home", "role", "tier", "purity", "w1",
            "w2", "top2", "second", "w_second", "snr_home", "n_mass",
            "purity_swing", "tier_stable", "neg_mass", "served_own",
            "r_res_own_init", "response_mean"]
    rows = []
    for arm in arms:
        for district in districts:
            for kind in ("pressure", "flow"):
                sub = _ranked(table, arm, district, kind, channels,
                              enforce_channel)
                head = dedupe(sub, n_max, min_distance,
                              on=ARMS[arm].dedupe_on)
                for r, (_, row) in enumerate(head.iterrows(), start=1):
                    rows.append({"config_id": config_id, "arm": arm,
                                 "district": district, "kind": kind, "rank": r,
                                 "n_eligible": int(len(sub)),
                                 **{c: row.get(c) for c in keep if c in row}})
    return pd.DataFrame(rows)


def coverage(table: pd.DataFrame, districts, config_id, arms=PIPELINE_ARMS,
             n_max: int = 5, channels: pd.DataFrame | None = None,
             enforce_channel: bool = True,
             min_distance: float = DEDUPE_MIN_DISTANCE) -> pd.DataFrame:
    """Per (arm, district, kind): how full the set is, and why it is not fuller.

    `n_eligible` against `n_selected` separates the two reasons a set is short.
    A district with `n_eligible` 0 has no gauge meeting the criteria. One with
    `n_eligible` 15 and `n_selected` 2 has plenty of gauges and only two
    distinct READINGS among them -- `redundancy` names that case, and it is the
    one that used to be invisible.
    """
    channels = channel_diversity(table) if channels is None else channels
    degenerate = set(channels.loc[channels["verdict"] != "ok", "kind"])
    rows = []
    for arm in arms:
        for district in districts:
            for kind in ("pressure", "flow"):
                sub = _ranked(table, arm, district, kind, channels,
                              enforce_channel)
                n = len(dedupe(sub, n_max, min_distance,
                               on=ARMS[arm].dedupe_on))
                rows.append({
                    "config_id": config_id, "arm": arm, "district": district,
                    "kind": kind, "n_selected": int(n),
                    "n_eligible": int(len(sub)), "n_max": int(n_max),
                    "redundancy": round(len(sub) / n, 2) if n else np.nan,
                    "channel": ("degenerate" if kind in degenerate else "ok"),
                    "complete": bool(n == n_max), "empty": bool(n == 0)})
    return pd.DataFrame(rows)


def sensors_yaml(sel: dict) -> str:
    """The selection in the shape a bundle's `profile.yml` wants."""
    return sp.sensors_yaml(sel)


# ==========================================================================
# the anti-phase variant -- a diagnostic, not a shipped path
# ==========================================================================
def rebuild_mixture(gains: pd.DataFrame, kind: str, mode: str = "positive",
                    null_factor: float = 1.5) -> pd.DataFrame:
    """Rebuild the simplex from stored gains, optionally keeping ANTI-PHASE mass.

    `mode="positive"` reproduces what `mixture_probe.build_mixture` does: a
    negative gain is dropped before normalising, so a gauge that moves hard in
    the opposite direction to a district's drift records no weight for it.

    `mode="magnitude"` normalises `|g|` instead. On a pump-and-tank network
    that is not a cosmetic difference. KY7 flow carries anti-phase mass on
    65.7% of gauges against D-Town's 39.7%, because trunk mains there reverse
    when a district draws harder -- and a reversal IS dependence, arguably the
    most physical kind. On a reconstruction from these gains, `magnitude`
    roughly doubles the mid-band population on KY7 spectral flow (0.148 against
    0.065) and leaves D-Town nearly unchanged (girvan_newman 22 against 19
    gauges through the target gate), which is the signature of recovering
    something real on one network and nothing spurious on the other.

    It is a DIAGNOSTIC and is NOT wired into the arms. Two reasons to prove it
    before trusting it. First, `dependence()` already averages `|E|`, so the
    simplex and the dependence matrix currently disagree about sign and
    `magnitude` would make them agree -- an argument FOR the change, but it is
    a change, and the numbers above come from a reconstruction rather than from
    `build_mixture` itself. Second, folding the sign in loses it, so
    `phase_sign` and `neg_live` are carried out separately: a gauge blending
    +0.6 of A with -0.4 of B is a different physical object from one blending
    +0.6 and +0.4, and the FL step should be able to tell them apart.

    Validate it by running `channel_diversity` on the magnitude simplex before
    the positive one. If magnitude's extra mid-band gauges are all the same
    reading, it recovered nothing.
    """
    if mode not in ("positive", "magnitude"):
        raise ValueError("mode must be 'positive' or 'magnitude'")
    G = gains[gains["kind"] == kind]
    if not len(G):
        return pd.DataFrame()

    def piv(col):
        return G.pivot_table(index="id", columns="drift_district", values=col,
                             aggfunc="first")

    P, N, E = piv("g"), piv("null"), piv("excite")
    thr = null_factor * (N / E)
    live = (P.abs() if mode == "magnitude" else P) > thr
    V = (P.abs() if mode == "magnitude" else P.clip(lower=0)).where(live, 0.0)
    tot = V.sum(axis=1)
    W = V.div(tot.replace(0.0, np.nan), axis=0)
    S = np.sort(W.fillna(0.0).to_numpy(float), axis=1)

    out = W.add_prefix("w_").copy()
    out["kind"] = kind
    out["mode"] = mode
    out["n_live"] = live.sum(axis=1)
    out["mass"] = tot
    out["w1"], out["w2"] = S[:, -1], S[:, -2]
    out["top2"] = S[:, -1] + S[:, -2]
    # the sign that `magnitude` folds away, kept so nothing is lost
    signed = P.where(live, np.nan)
    out["phase_sign"] = np.sign(signed.mean(axis=1)).fillna(0.0)
    out["neg_live"] = (signed < 0).sum(axis=1)
    return out.reset_index()
