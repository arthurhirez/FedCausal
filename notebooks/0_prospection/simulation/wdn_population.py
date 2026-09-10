"""Population and land-use demand generation on an hourly clock.

The causal chain
----------------
    population -> per-capita demand -> base demand -> pattern shape by land use
    -> hydraulics

    d_j(t) = P_j(t) * q(income_j(t)) * sum_c w_jc(t) * pe_c * shape_c(t)
             * seasonal(t) * noise_d(t)

Population is the state variable and pressure is an outcome. Demand grows
because there are more people; whether the network copes is measured, not
assumed.

Why this evaluates hourly rather than in chunks
-----------------------------------------------
Chunked execution exists to avoid per-junction full-horizon patterns. At the
scale of this network that cost was measured rather than assumed: 348 consumers
x 28 800 hourly steps is a 106 MB scratch .inp that EPANET solves in about 52 s
in one shot. Chunking is therefore unnecessary here, and avoiding it buys four
things that are not stylistic:

1. **Volume stays exact by construction.** The output is the L/s series itself;
   the hydraulics stage sets ``base_value = 1 L/s`` and makes the *pattern* the
   series, so no rescaling step can drift.
2. **No chunk-boundary artefacts.** Restarting EPANET per chunk and carrying
   tank levels freezes every observable for exactly one step at every boundary.
   Measured: mean |delta| across a boundary was 0.0000 for tank level and pump
   status against 0.3734 and 0.1083 in the interior -- a duplicated state, once
   per chunk, at perfectly regular spacing. Transfer entropy reads exactly those
   consecutive pairs, and a changepoint detector would find the boundaries.
   Running monolithically, tank and pump state is continuous because the solver
   never restarts.
3. **Hourly drift resolution.** A per-node linear blend over ``ramp_hours``
   works directly. Under chunking the same drift quantises to the chunk grid.
4. **No per-chunk volume leak.** Whole-horizon mean-1 noise normalisation is
   simply correct when there are no chunks to slice it into.

The threshold matters if this moves to a bigger corpus: both the file size and
the solve time scale roughly as nodes x steps, so ky17 (2 622 consumers) or a
ten-year horizon brings chunking back. That is the only reason to reach for it.

Three slow axes, and an identifiability warning
-----------------------------------------------
Population and per-capita demand **multiply**, so a 20% population rise and a
20% income-driven rise in consumption are indistinguishable from water data
alone: same level, same shape. Land use is different, because it changes the
*shape*, and shape is identifiable from level. Therefore land use is recoverable
by an observational estimator; population and income are recoverable only as
their product. ``ScenarioConfig.assert_identifiable`` rejects scenarios that
drift shape and level in overlapping windows on shared nodes.

Drift geometry is node-level
----------------------------
Drift spreads by diffusion from a seed node over the affected district's
subgraph, giving each node its own switch time. This is deliberate and is not
the same experiment as a district-level switch: with per-node timing, a client
is *partially* drifted for a stretch of the horizon, so "how much of District_A
is affected" is a quantity an estimator can be wrong about. A district-level
switch collapses drift to a per-client binary label, which makes detection
easier and removes within-client heterogeneity as a source of signal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from wdn_capacity import (LPS_PER_M3S, SEC_PER_DAY, DemandModelConfig,
                          base_demand_of, consumer_nodes, pattern_mean_of)
from wdn_demand import CATEGORIES, ShapeParams

DRIFT_KINDS = ("landuse_conversion", "growth_rate", "income_shift")


# ==========================================================================
# clock
# ==========================================================================

@dataclass
class HorizonConfig:
    """The hourly clock, matching the existing engine's month convention."""
    n_months: int = 40
    days_per_month: int = 30
    timestep_s: int = 3600
    start_dow: int = 0          # 0 = Monday
    start_month_of_year: int = 0    # 0 = January

    @property
    def steps_per_month(self) -> int:
        return int(self.days_per_month * 24 * 3600 // self.timestep_s)

    @property
    def n_steps(self) -> int:
        return self.n_months * self.steps_per_month

    def validate(self):
        if 3600 % self.timestep_s and self.timestep_s % 3600:
            raise ValueError("timestep_s should divide or be a multiple of 1 h")
        if self.n_months < 2:
            raise ValueError("n_months must be >= 2")
        return self

    def calendar(self) -> pd.DataFrame:
        """Per-step calendar features.

        ``month`` is the integer month index the existing engine keys on.
        ``month_continuous`` is its fractional counterpart, used for seasonality
        so the annual cycle is smooth rather than a monthly staircase.
        """
        step_h = self.timestep_s / 3600.0
        t = np.arange(self.n_steps) * step_h
        day = (t // 24).astype(int)
        mc = t / (self.days_per_month * 24.0)
        return pd.DataFrame({
            "t_h": t,
            "hour": (t % 24).astype(int),
            "day": day,
            "dow": (day + self.start_dow) % 7,
            "month": np.minimum((mc).astype(int), self.n_months - 1),
            "month_continuous": mc + self.start_month_of_year,
        })


# ==========================================================================
# assignment
# ==========================================================================

@dataclass
class AssignmentConfig:
    """How junctions get land-use mixes, income bands and population shares.

    ``anchor`` decides where the *spatial* distribution of demand comes from,
    and it is the setting most likely to invalidate a result:

    ``demand``  -- month-0 demand per node reproduces the .inp's own mean demand
                   (base x pattern mean). The network was sized against that
                   distribution -- diameters, tank siting, pump curves, control
                   setpoints -- so this keeps chunk 0 at the design point and
                   makes everything after it attributable to growth.
    ``served_area`` -- population proportional to half the incident pipe length.
                   Measured on D-Town: this is **uncorrelated** with the .inp
                   base demand (r = -0.01), misallocates by up to 1543x at a
                   node, and costs 7.5 m of minimum pressure at identical total
                   demand. Those deficits are artefacts of the heuristic, not of
                   population. Available for ablation, not recommended.
    """
    target_mix: dict = field(default_factory=lambda: {
        "residential": 0.70, "commercial": 0.22, "industrial": 0.08})
    mix_concentration: float = 6.0
    industrial_demand_quantile: float = 0.90
    income_bands: tuple = ("low", "medium", "high")
    income_weights: tuple = (0.45, 0.40, 0.15)
    anchor: str = "demand"
    seed: int = 0

    def validate(self):
        if abs(sum(self.target_mix.values()) - 1.0) > 1e-6:
            raise ValueError("target_mix must sum to 1")
        if set(self.target_mix) - set(CATEGORIES):
            raise ValueError(f"target_mix keys must be within {CATEGORIES}")
        if len(self.income_bands) != len(self.income_weights):
            raise ValueError("income_bands and income_weights must align")
        if self.anchor not in ("demand", "served_area"):
            raise ValueError("anchor must be 'demand' or 'served_area'")
        return self


def served_area_proxy(wn, nodes) -> pd.Series:
    """Half the length of incident pipes: the standard served-area proxy."""
    out = {}
    for n in nodes:
        L = 0.0
        for lname in wn.get_links_for_node(n):
            l = wn.get_link(lname)
            if l.link_type == "Pipe":
                L += 0.5 * float(l.length)
        out[n] = L
    return pd.Series(out, name="served_length_m")


def assign_landuse(wn, districts: pd.Series, cfg: AssignmentConfig,
                   dm: DemandModelConfig | None = None,
                   classifier=None) -> pd.DataFrame:
    """Assign land-use mix, income band and population share to each consumer.

    ``districts`` is required and is the caller's partition -- typically the
    utility's DMA map read from YAML. There is deliberately no clustering
    fallback: a partition inferred from coordinates would silently disagree with
    the sensor placement and the district boundaries the rest of the study is
    built on.

    Zones get a mix drawn around the corpus target; junctions then get a
    Dirichlet draw around their zone's mix, so neighbours resemble each other
    without being identical. That produces spatial structure in the *shape* of
    demand, which is what a land-use drift later perturbs.

    Pass ``classifier(wn, nodes, cfg) -> DataFrame`` to substitute real land-use
    or income layers.
    """
    cfg = cfg.validate()
    dm = dm or DemandModelConfig()
    nodes = [n for n in consumer_nodes(wn) if n in districts.index]
    if not nodes:
        raise ValueError("no consumer junctions found in the district partition")

    if classifier is not None:
        out = classifier(wn, nodes, cfg)
        missing = {"district", "income", *CATEGORIES} - set(out.columns)
        if missing:
            raise ValueError(f"classifier output missing columns: {sorted(missing)}")
        return out

    rng = np.random.default_rng(cfg.seed)
    area = served_area_proxy(wn, nodes)
    base = pd.Series({n: base_demand_of(wn, n) for n in nodes})
    pmean = pd.Series({n: pattern_mean_of(wn, n) for n in nodes})
    inp_mean = base * pmean                      # true mean demand, m3/s

    target = np.array([cfg.target_mix.get(c, 0.0) for c in CATEGORIES])
    zone_mix = {}
    for d in sorted(pd.unique(districts.reindex(nodes).dropna())):
        zone_mix[d] = rng.dirichlet(np.maximum(target, 1e-3) * cfg.mix_concentration)

    ind_cut = base.quantile(cfg.industrial_demand_quantile) if len(base) else np.inf
    rows = []
    for n in nodes:
        d = districts.get(n)
        m = zone_mix.get(d, target).copy()
        # large existing demand leans industrial/commercial: a proxy for the
        # fact that big draw-offs are rarely single households
        if base.get(n, 0.0) >= ind_cut:
            m = m + np.array([0.0, 0.10, 0.25])
        w = rng.dirichlet(np.maximum(m, 1e-3) * cfg.mix_concentration)
        band = rng.choice(cfg.income_bands,
                          p=np.asarray(cfg.income_weights, dtype=float)
                          / sum(cfg.income_weights))
        rows.append({"node": n, "district": d, "income": band,
                     "served_length_m": float(area.get(n, 0.0)),
                     "inp_base_lps": float(base.get(n, 0.0)) * LPS_PER_M3S,
                     "inp_pattern_mean": float(pmean.get(n, 1.0)),
                     "inp_mean_lps": float(inp_mean.get(n, 0.0)) * LPS_PER_M3S,
                     **dict(zip(CATEGORIES, w))})
    out = pd.DataFrame(rows).set_index("node")

    # per-capita demand intensity at month 0, m3/s per population unit
    q = out.income.map(dm.income_factors).astype(float) \
        * dm.litres_per_capita_day / 1000.0 / SEC_PER_DAY
    pe = sum(out[c] * dm.pe_per_unit[c] for c in CATEGORIES)
    out["intensity_m3s_per_pe"] = q * pe

    if cfg.anchor == "demand":
        # population share chosen so that month-0 *demand* per node reproduces
        # the .inp distribution, after dividing out each node's own intensity
        tgt = out.inp_mean_lps.to_numpy()
        if tgt.sum() <= 0:
            raise ValueError("anchor='demand' needs non-zero .inp base demands")
        share = tgt / out.intensity_m3s_per_pe.to_numpy()
    else:
        s = out.served_length_m
        share = s.clip(lower=s[s > 0].min() if (s > 0).any() else 1.0).to_numpy()
    out["pop_share"] = share / share.sum()
    return out


# ==========================================================================
# drift: node-level diffusion geometry
# ==========================================================================

@dataclass
class DriftSpec:
    """One drift primitive, spread over nodes by diffusion from a seed.

    ``kind``:
      ``landuse_conversion`` -- move mix weight from one category to another;
        changes the **shape** of demand, identifiable from level.
      ``growth_rate``        -- multiply the affected nodes' growth rate;
        changes the **level**.
      ``income_shift``       -- raise the income factor; changes the level only,
        and is therefore not separable from growth by design.

    Timing: node hop distance from ``seed_node`` maps linearly onto
    ``[start_month, end_month]``, and each node then blends over ``ramp_days``.
    ``seed_node=None`` seeds at the affected district's largest .inp demand -- a
    development centre, chosen deterministically so the geometry is reproducible.
    """
    kind: str
    districts: tuple = ()
    seed_node: str | None = None
    start_month: int = 2
    end_month: int = 19
    ramp_days: float = 10.0
    magnitude: float = 0.0
    from_category: str = "residential"
    to_category: str = "industrial"

    def validate(self, n_months: int):
        if self.kind not in DRIFT_KINDS:
            raise ValueError(f"unknown drift kind {self.kind!r}")
        if not 0 <= self.start_month < n_months:
            raise ValueError("start_month outside the horizon")
        if self.end_month < self.start_month:
            raise ValueError("end_month must be >= start_month")
        if self.end_month >= n_months:
            raise ValueError("end_month outside the horizon")
        if self.ramp_days <= 0:
            raise ValueError("ramp_days must be > 0")
        if not self.districts:
            raise ValueError(
                "districts is empty, which would drift every district at once. "
                "That is a global common-mode shift and is undetectable as an "
                "anomaly by construction. Name the affected districts.")
        if self.kind == "landuse_conversion":
            for c in (self.from_category, self.to_category):
                if c not in CATEGORIES:
                    raise ValueError(f"unknown category {c!r}")
            if not 0.0 <= self.magnitude <= 1.0:
                raise ValueError("conversion magnitude must be in [0, 1]")
        return self

    @property
    def window(self) -> tuple[float, float]:
        """First switch to last switch complete, in months."""
        return self.start_month, self.end_month + self.ramp_days / 30.0


def _hop_distances(wn, nodes: set, seed: str) -> tuple[dict, set]:
    """BFS hop distance from ``seed``, restricted to ``nodes``.

    Restricting to the district's own nodes is what makes this a *district*
    diffusion: development spreads through the neighbourhood, not through a
    trunk main that happens to pass by. Unreachable nodes are placed one hop
    beyond the farthest reachable one, so they switch last rather than never.
    """
    dist = {seed: 0}
    dq = deque([seed])
    while dq:
        u = dq.popleft()
        for lname in wn.get_links_for_node(u):
            l = wn.get_link(lname)
            for v in (l.start_node_name, l.end_node_name):
                if v in nodes and v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
    reached = set(dist)
    far = max(dist.values()) if dist else 0
    for n in nodes:
        dist.setdefault(n, far + 1)
    return dist, reached


def build_drift_schedule(wn, assignment: pd.DataFrame, drifts: tuple,
                         horizon: HorizonConfig,
                         districts: pd.Series | None = None) -> pd.DataFrame:
    """Per-node switch time and ramp for every drift. This is ground truth.

    One row per (drift, affected node). Nodes outside the affected districts get
    no row, which is what makes "unaffected" checkable rather than implied.

    ``districts`` is the **full** partition, including zero-demand junctions and
    storage nodes. It is needed for the traversal even though only consumers get
    rows: those junctions carry no demand but they are physically present, and
    development spreads through them. Restricting the BFS to consumers alone
    fragments a district that is in fact connected -- measured on D-Town,
    District_A splits into an unreachable majority (62 of 93 consumers) because
    its zero-demand junctions are the articulation points.
    """
    horizon = horizon.validate()
    spm = horizon.steps_per_month
    rows = []
    for di, spec in enumerate(drifts):
        spec = spec.validate(horizon.n_months)
        aff = assignment[assignment.district.isin(spec.districts)]
        if aff.empty:
            raise ValueError(f"drift {di} ({spec.kind}) affects no nodes: "
                             f"districts {spec.districts} not in the partition")
        if spec.seed_node is not None and spec.seed_node not in aff.index:
            raise ValueError(f"seed_node {spec.seed_node!r} is not in the "
                             f"affected districts {spec.districts}")

        # Diffuse within each district separately. Districts need not be
        # adjacent -- on D-Town, A and C touch only through District_E's pumps
        # -- so a single BFS over the union would leave one district entirely
        # unreachable and collapse all of its nodes onto the fallback switch
        # time. Each district gets its own development centre instead.
        for d in spec.districts:
            sub = aff[aff.district == d]
            if sub.empty:
                continue
            seed = spec.seed_node if (spec.seed_node in sub.index) else \
                sub.inp_mean_lps.idxmax()
            if districts is not None:
                walk = set(districts.index[districts == d])
            else:
                import warnings as _w
                _w.warn("districts not supplied; diffusing over consumer nodes "
                        "only, which can fragment a connected district")
                walk = set(sub.index)
            hops, reached = _hop_distances(wn, walk, seed)
            h = pd.Series(hops).reindex(sub.index).astype(float)
            missed = [n for n in sub.index if n not in reached]
            if missed:
                import warnings as _w
                _w.warn(
                    "%s: %d of %d consumers unreachable from seed %s within the "
                    "district, so they switch last rather than in diffusion "
                    "order. The district is internally disconnected."
                    % (d, len(missed), len(sub), seed))
            span = h.max() - h.min()
            frac = (h - h.min()) / span if span > 0 else h * 0.0
            switch_month = spec.start_month \
                + frac * (spec.end_month - spec.start_month)
            for n in sub.index:
                rows.append({
                    "drift_id": di, "kind": spec.kind, "node": n,
                    "district": d, "seed_node": seed, "hop": int(h[n]),
                    "reachable": bool(n in reached),
                    "switch_month": float(switch_month[n]),
                    "switch_step": float(switch_month[n] * spm),
                    "ramp_steps": float(spec.ramp_days * 24 * 3600
                                        / horizon.timestep_s),
                    "magnitude": spec.magnitude,
                    "from_category": spec.from_category,
                    "to_category": spec.to_category,
                })
    return pd.DataFrame(rows)


def _ramp(n_steps: int, switch_step: float, ramp_steps: float) -> np.ndarray:
    """Linear 0 -> 1 blend, matching the existing engine's 10-day ramp."""
    k = np.arange(n_steps, dtype=float)
    return np.clip((k - switch_step) / max(ramp_steps, 1e-9), 0.0, 1.0)


# ==========================================================================
# growth and noise
# ==========================================================================

@dataclass
class GrowthConfig:
    """Logistic population growth, heterogeneous across districts."""
    initial_capacity_fraction: float = 0.40
    logistic_rate_per_month: float = 0.045
    logistic_ceiling_fraction: float = 0.95
    district_rate_spread: float = 0.5
    seed: int = 0


@dataclass
class NoiseConfig:
    """Multiplicative AR(1) noise, correlated within a district.

    Two components: a system-wide term (weather, holidays, anything that moves
    every district at once) and a district-specific term. Independent
    per-junction noise is deliberately absent -- it would average out over a
    district and so could not create the cross-client dependence the federated
    setting is about.

    Persistence is specified as a **correlation time in hours**, not as a
    per-step coefficient. The distinction matters: a coefficient of 0.4 applied
    at hourly resolution decorrelates in about an hour, which is white noise for
    any purpose here, whereas a district-level driver such as weather persists
    for days. ``tau_h`` makes that assumption explicit and resolution-independent.
    """
    sigma_district: float = 0.06
    sigma_common: float = 0.03
    tau_h: float = 48.0
    seed: int = 0

    def phi(self, timestep_s: int) -> float:
        dt_h = timestep_s / 3600.0
        return float(np.exp(-dt_h / max(self.tau_h, 1e-9)))


def district_noise(districts, horizon: HorizonConfig,
                   cfg: NoiseConfig) -> pd.DataFrame:
    """Mean-1 multiplicative noise per district over the whole horizon."""
    rng = np.random.default_rng(cfg.seed + 977)
    T = horizon.n_steps
    phi = cfg.phi(horizon.timestep_s)

    def ar1(sigma):
        if sigma <= 0:
            return np.zeros(T)
        e = rng.normal(0.0, sigma * np.sqrt(1 - phi ** 2), size=T)
        x = np.empty(T)
        x[0] = rng.normal(0.0, sigma)
        for t in range(1, T):
            x[t] = phi * x[t - 1] + e[t]
        return x

    common = ar1(cfg.sigma_common)
    out = {}
    for d in districts:
        v = np.exp(common + ar1(cfg.sigma_district))
        out[d] = (v / v.mean()).astype(np.float32)
    return pd.DataFrame(out)


def _population_paths(assignment: pd.DataFrame, p_max: float,
                      horizon: HorizonConfig, growth: GrowthConfig,
                      rate_mult: np.ndarray) -> np.ndarray:
    """Per-node logistic population on the monthly grid, shape (n_months, N).

    The logistic is per node rather than per district so that a node-level
    growth drift acts where its geometry says it does; the district path is the
    sum of its nodes.
    """
    rng = np.random.default_rng(growth.seed)
    nodes = list(assignment.index)
    dists = sorted(pd.unique(assignment.district))
    d_rate = {d: growth.logistic_rate_per_month
              * float(rng.uniform(1 - growth.district_rate_spread,
                                  1 + growth.district_rate_spread))
              for d in dists}
    rate0 = assignment.district.map(d_rate).to_numpy(dtype=float)
    share = assignment.pop_share.to_numpy(dtype=float)

    M, N = horizon.n_months, len(nodes)
    cap = growth.logistic_ceiling_fraction * p_max * share
    P = np.empty((M, N))
    P[0] = growth.initial_capacity_fraction * p_max * share
    for m in range(1, M):
        r = rate0 * rate_mult[m]
        with np.errstate(invalid="ignore", divide="ignore"):
            grow = np.where(cap > 0, r * P[m - 1] * (1 - P[m - 1] / cap), 0.0)
        P[m] = P[m - 1] + grow
    return P, pd.Series(d_rate, name="rate_per_month")


# ==========================================================================
# the scenario
# ==========================================================================

@dataclass
class ScenarioConfig:
    """Everything that defines one world."""
    horizon: HorizonConfig = field(default_factory=HorizonConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    seasonal: bool = True
    drifts: tuple = ()
    allow_unidentifiable: bool = False

    def validate(self):
        self.horizon.validate()
        for d in self.drifts:
            if isinstance(d, dict):
                raise TypeError(
                    "drifts contains a dict, not a DriftSpec. This happens when "
                    "a config is rebuilt from dataclasses.asdict(), which "
                    "converts nested dataclasses recursively. Use "
                    "dataclasses.replace(cfg, field=value) instead.")
            d.validate(self.horizon.n_months)
        return self

    def assert_identifiable(self):
        """Reject scenarios whose ground truth cannot be recovered.

        Land use is identifiable from shape; population and income are only
        identifiable as a product. If a level drift overlaps a shape drift on
        shared districts, an observational estimator cannot attribute the
        change and any recovery claim would be unfalsifiable.
        """
        shape = [d for d in self.drifts if d.kind == "landuse_conversion"]
        level = [d for d in self.drifts if d.kind in ("growth_rate", "income_shift")]
        clashes = []
        for a in shape:
            for b in level:
                a0, a1 = a.window
                b0, b1 = b.window
                if a0 < b1 and b0 < a1 and (set(a.districts) & set(b.districts)):
                    clashes.append((a.kind, b.kind, (a0, a1), (b0, b1)))
        if clashes and not self.allow_unidentifiable:
            raise ValueError(
                "shape drift and level drift overlap in time on shared "
                f"districts, so the ground truth is not recoverable: {clashes}. "
                "Separate the windows, or set allow_unidentifiable=True to "
                "proceed knowingly.")
        return clashes


def anchor_population(assignment: pd.DataFrame, capacity_lps: float,
                      dm: DemandModelConfig,
                      apply_peak_division: bool = False) -> dict:
    """Population ceiling implied by the capacity, using the real assignment.

    ``capacity_to_population`` in ``wdn_capacity`` answers the same question
    from a mean mix and a single income band, which is the right shape for a
    report. Here we have the actual per-node mixes and bands, so the ceiling is
    computed from them and is exact rather than approximate.
    """
    inten = assignment.intensity_m3s_per_pe.to_numpy(dtype=float)
    share = assignment.pop_share.to_numpy(dtype=float)
    per_pe = float((share * inten).sum())          # m3/s of demand per unit P
    if per_pe <= 0:
        raise ValueError("assignment has zero demand intensity")
    peak = dm.k1_daily_peak * dm.k2_hourly_peak if apply_peak_division else 1.0
    p_max = capacity_lps / LPS_PER_M3S / peak / per_pe
    return {"population_capacity": p_max,
            "demand_per_population_unit_lps": per_pe * LPS_PER_M3S,
            "peak_divisor_applied": peak,
            "target_mean_lps_at_capacity": capacity_lps / peak}


def synthesize_demands(wn, assignment: pd.DataFrame, schedule: pd.DataFrame,
                       shapes: ShapeParams, dm: DemandModelConfig,
                       cfg: ScenarioConfig, p_max: float,
                       verbose: bool = True) -> dict:
    """Hourly demand in L/s per node, plus the ground truth that produced it.

    ``demand_series`` is the artifact the hydraulics stage consumes: a
    ``month`` column followed by one column per node, in L/s. Volume is exact by
    construction because this series *is* the demand -- the hydraulics stage
    makes it the pattern and sets the base value to 1 L/s, so there is no
    rescaling step that can drift.
    """
    cfg = cfg.validate()
    cfg.assert_identifiable()
    hz, T = cfg.horizon, cfg.horizon.n_steps
    cal = hz.calendar()
    nodes = list(assignment.index)
    dists = sorted(pd.unique(assignment.district))
    spm = hz.steps_per_month
    if verbose:
        print("horizon %d months x %d days = %d steps, %d nodes"
              % (hz.n_months, hz.days_per_month, T, len(nodes)))

    # ---- shared shapes: mean-1 over the whole horizon ---------------------
    hour, dow = cal.hour.to_numpy(), cal.dow.to_numpy()
    shape = {}
    for c in CATEGORIES:
        s = shapes.diurnal(c)[hour] * shapes.weekly(c)[dow]
        shape[c] = (s / s.mean()).astype(np.float64)

    # Seasonality is evaluated on the *continuous* month, so the annual cycle is
    # a smooth sinusoid rather than a monthly staircase. It also comes from the
    # ``shapes`` object that was passed in -- reinstantiating ShapeParams here
    # would silently ignore a caller's seasonal_amplitude.
    seas = shapes.seasonal(cal.month_continuous.to_numpy()) if cfg.seasonal \
        else np.ones(T)

    noise = district_noise(dists, hz, cfg.noise)

    # ---- per-node drift ramps -------------------------------------------
    idx = {n: i for i, n in enumerate(nodes)}
    conv = {}                       # (node, from, to) -> ramp array
    rate_mult = np.ones((hz.n_months, len(nodes)))
    inc_mult = {}
    for r in schedule.itertuples(index=False):
        if r.node not in idx:
            continue
        if r.kind == "landuse_conversion":
            ramp = _ramp(T, r.switch_step, r.ramp_steps)
            key = (r.node, r.from_category, r.to_category)
            conv[key] = conv.get(key, 0.0) + r.magnitude * ramp
        elif r.kind == "growth_rate":
            mr = _ramp(hz.n_months, r.switch_step / spm, r.ramp_steps / spm)
            rate_mult[:, idx[r.node]] *= (1.0 + r.magnitude * mr)
        elif r.kind == "income_shift":
            ramp = _ramp(T, r.switch_step, r.ramp_steps)
            inc_mult[r.node] = inc_mult.get(r.node, 0.0) + r.magnitude * ramp

    P_month, growth_rates = _population_paths(assignment, p_max, hz,
                                              cfg.growth, rate_mult)
    # linear interpolation onto the hourly clock: smooth growth, no staircase
    m_at = cal.t_h.to_numpy() / (hz.days_per_month * 24.0)
    grid = np.arange(hz.n_months, dtype=float)

    # ---- assemble ---------------------------------------------------------
    dem = np.empty((T, len(nodes)), dtype=np.float32)
    mix_rows = []
    for i, n in enumerate(nodes):
        pop = np.interp(m_at, grid, P_month[:, i])
        w = {c: np.full(T, float(assignment.at[n, c])) for c in CATEGORIES}
        for (nn, cf, ct), ramp in conv.items():
            if nn != n:
                continue
            moved = np.full(T, float(assignment.at[n, cf])) * ramp
            w[cf] = w[cf] - moved
            w[ct] = w[ct] + moved
        tot = sum(w.values())
        acc = np.zeros(T)
        for c in CATEGORIES:
            w[c] = np.clip(w[c], 0.0, None) / np.where(tot > 0, tot, 1.0)
            acc += w[c] * dm.pe_per_unit[c] * shape[c]
        f_inc = float(dm.income_factors[assignment.at[n, "income"]])
        inc = f_inc * (1.0 + np.asarray(inc_mult.get(n, 0.0)) * np.ones(T))
        q = inc * dm.litres_per_capita_day / 1000.0 / SEC_PER_DAY
        d = assignment.at[n, "district"]
        dem[:, i] = (pop * q * acc * seas
                     * noise[d].to_numpy(dtype=float)) * LPS_PER_M3S
        for m in range(hz.n_months):
            sl = slice(m * spm, (m + 1) * spm)
            mix_rows.append({"month": m, "node": n, "district": d,
                             "population": float(pop[sl].mean()),
                             "income_factor": float(inc[sl].mean()),
                             **{c: float(w[c][sl].mean()) for c in CATEGORIES}})

    demand_series = pd.DataFrame(dem, columns=nodes)
    demand_series.insert(0, "month", cal.month.to_numpy())
    trajectory = pd.DataFrame(mix_rows)

    if verbose:
        tot = dem.sum(axis=1)
        print("mean %.1f L/s, peak %.1f L/s (peak factor %.2f), month-0 mean %.1f L/s"
              % (tot.mean(), tot.max(), tot.max() / tot.mean(), tot[:spm].mean()))

    return {"demand_series": demand_series,
            "trajectory": trajectory,
            "calendar": cal,
            "noise": noise,
            "growth_rates": growth_rates,
            "population_capacity": p_max,
            "config": {"horizon": asdict(cfg.horizon),
                       "growth": asdict(cfg.growth),
                       "noise": asdict(cfg.noise),
                       "seasonal": cfg.seasonal}}


# ==========================================================================
# validation
# ==========================================================================

def validate_demands(demand_series: pd.DataFrame, assignment: pd.DataFrame,
                     capacity_lps: float, horizon: HorizonConfig) -> pd.DataFrame:
    """Per-month realised load against capacity, before any solve.

    Validating only the initial state is the trap: growth is monotone, so a
    scenario that starts comfortably can end far past the network's limit and
    would waste the whole run before failing. Peak here is the *realised*
    hourly peak of the actual series, not a nominal peak factor.
    """
    nodes = [c for c in demand_series.columns if c != "month"]
    tot = demand_series[nodes].to_numpy().sum(axis=1)
    m = demand_series.month.to_numpy()
    rows = []
    for k in range(horizon.n_months):
        s = tot[m == k]
        if not s.size:
            continue
        rows.append({"month": k, "mean_lps": float(s.mean()),
                     "peak_lps": float(s.max()),
                     "peak_factor_realised": float(s.max() / s.mean())})
    out = pd.DataFrame(rows)
    out["capacity_lps"] = capacity_lps
    out["headroom_ratio"] = capacity_lps / out.mean_lps
    out["breach"] = out.mean_lps > capacity_lps
    return out


def check_anchor(demand_series: pd.DataFrame, assignment: pd.DataFrame,
                 horizon: HorizonConfig) -> pd.DataFrame:
    """Does month 0 reproduce the .inp's spatial demand distribution?

    The single check that catches an anchoring mistake. If ``anchor='demand'``
    worked, the per-node correlation is ~1.0 and the worst node ratio is ~1.0.
    Under ``anchor='served_area'`` the correlation collapses -- measured -0.01
    on D-Town.
    """
    nodes = [c for c in demand_series.columns if c != "month"]
    spm = horizon.steps_per_month
    m0 = demand_series.loc[:spm - 1, nodes].mean()
    tgt = assignment.inp_mean_lps.reindex(nodes)
    ratio = (m0 / tgt.replace(0, np.nan)) / (m0.sum() / tgt.sum())
    return pd.DataFrame({
        "metric": ["pearson_r", "spearman_r", "ratio_p01", "ratio_p99",
                   "total_month0_lps", "total_inp_mean_lps"],
        "value": [float(m0.corr(tgt)), float(m0.corr(tgt, method="spearman")),
                  float(ratio.quantile(0.01)), float(ratio.quantile(0.99)),
                  float(m0.sum()), float(tgt.sum())]})
