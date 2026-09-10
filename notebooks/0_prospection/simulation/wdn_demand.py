"""Population and land-use demand generation, and chunked long-horizon simulation.

The causal chain
----------------
    population -> per-capita demand -> base demand -> pattern shape by land use
    -> hydraulics

Population is the state variable and pressure is an outcome. This matters: the
earlier approach of scaling a global demand multiplier until pressure hits its
floor bakes the answer into the setup, because it redistributes consumption to
land exactly on the limit. Here demand *grows* because there are more people,
and whether the network copes is something we measure.

Three slow axes, and an identifiability warning
-----------------------------------------------
    d_j(t) = P_j(t) * q(income_j(t)) * sum_c w_jc(t) * shape_c(t) * seasonal(t) * noise(t)

* ``P_j``   population (or population equivalents) at junction j
* ``q``     litres per person per day, a function of income band
* ``w_jc``  land-use mix on the simplex; changes the **shape** of consumption
* ``shape_c`` the diurnal/weekly profile of land-use category c

Population and per-capita demand **multiply**, so from water data alone a 20%
population rise and a 20% income-driven rise in consumption are
indistinguishable: same level, same shape. Land use is different, because it
changes the shape, and shape is identifiable from level. Therefore:

* land use is recoverable by an observational estimator;
* population and income are recoverable only as their product.

``ScenarioConfig.assert_identifiable`` enforces the consequence: if a scenario
drifts land use and level in overlapping windows, the ground truth is not
recoverable and the scenario is rejected unless the caller explicitly opts out.
A separate ``connections`` field is carried alongside demand, because meter or
connection counts are the extra observable that would separate population from
income in a real utility, and it cannot be retrofitted after runs exist.

Why chunked execution is not just a memory trick
------------------------------------------------
Time-varying per-junction mixes would appear to need per-junction patterns over
the full horizon (ky17: 2 622 consumers x ~27 000 steps, which is unusable).
But within a chunk, population, income and mix are all constant, so a chunk
needs only a handful of shared patterns plus per-junction *base values*. Slow
drift lives in base values across chunks; fast shape lives in shared patterns.
The chunking a 40-month horizon forces on us for memory reasons is exactly the
mechanism that makes land-use drift cheap.

Noise is district-level by construction. Per-junction independent noise would
both explode the pattern count and average out across a district, destroying
the very signal a federated client is supposed to observe.
"""
from __future__ import annotations

import copy
import os
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import wntr

LPS_PER_M3S = 1000.0
SEC_PER_DAY = 86400.0
CATEGORIES = ("residential", "commercial", "industrial")


# ==========================================================================
# land-use shapes -- parameterised, never hardcoded arrays
# ==========================================================================

@dataclass
class ShapeParams:
    """Diurnal and weekly profile parameters per land-use category.

    Shapes are *generated* from interpretable parameters rather than pasted as
    magic arrays, so a reviewer can see what is being claimed and change it.
    Defaults follow the qualitative consensus in the water-demand literature:
    residential demand is bimodal (morning and evening peaks), commercial is a
    business-hours plateau, industrial is nearly flat with shift transitions.
    Every generated shape is normalised to mean 1, so the shape carries rhythm
    only and all volume information lives in the base demand.
    """
    # residential: two Gaussian peaks over a baseline
    res_morning_peak_h: float = 7.5
    res_evening_peak_h: float = 19.5
    res_peak_width_h: float = 2.2
    res_morning_amp: float = 1.0
    res_evening_amp: float = 1.3
    res_baseline: float = 0.35

    # commercial: logistic on/off plateau
    com_open_h: float = 8.0
    com_close_h: float = 18.0
    com_ramp_h: float = 1.0
    com_baseline: float = 0.15
    com_lunch_dip: float = 0.12
    com_lunch_h: float = 13.0

    # industrial: near-flat with shift steps
    ind_baseline: float = 0.85
    ind_shift_hours: tuple[float, ...] = (6.0, 14.0, 22.0)
    ind_shift_amp: float = 0.18
    ind_shift_width_h: float = 1.0

    # weekly multipliers (Mon..Sun), normalised internally to mean 1
    res_weekly: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.03, 1.12, 1.10)
    com_weekly: tuple[float, ...] = (1.05, 1.05, 1.05, 1.05, 1.10, 0.70, 0.45)
    ind_weekly: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.80)

    # seasonal sinusoid on the annual cycle
    seasonal_amplitude: float = 0.12
    seasonal_peak_month: int = 1        # 1 = January (southern-hemisphere summer)

    steps_per_hour: int = 1

    def _grid(self) -> np.ndarray:
        n = int(24 * self.steps_per_hour)
        return (np.arange(n) + 0.5) / self.steps_per_hour

    def diurnal(self, category: str) -> np.ndarray:
        """Mean-1 diurnal shape for one category."""
        h = self._grid()
        if category == "residential":
            g1 = self.res_morning_amp * np.exp(
                -0.5 * ((h - self.res_morning_peak_h) / self.res_peak_width_h) ** 2)
            g2 = self.res_evening_amp * np.exp(
                -0.5 * ((h - self.res_evening_peak_h) / self.res_peak_width_h) ** 2)
            v = self.res_baseline + g1 + g2
        elif category == "commercial":
            up = 1.0 / (1.0 + np.exp(-(h - self.com_open_h) / self.com_ramp_h))
            dn = 1.0 / (1.0 + np.exp((h - self.com_close_h) / self.com_ramp_h))
            dip = self.com_lunch_dip * np.exp(
                -0.5 * ((h - self.com_lunch_h) / 0.8) ** 2)
            v = self.com_baseline + up * dn - dip
        elif category == "industrial":
            v = np.full_like(h, self.ind_baseline)
            for s in self.ind_shift_hours:
                v += self.ind_shift_amp * np.exp(
                    -0.5 * ((h - s) / self.ind_shift_width_h) ** 2)
        else:
            raise ValueError(f"unknown category {category!r}")
        v = np.clip(v, 1e-6, None)
        return v / v.mean()

    def weekly(self, category: str) -> np.ndarray:
        w = np.asarray({"residential": self.res_weekly,
                        "commercial": self.com_weekly,
                        "industrial": self.ind_weekly}[category], dtype=float)
        if w.size != 7:
            raise ValueError("weekly multipliers must have 7 entries")
        return w / w.mean()

    def seasonal(self, month_index: np.ndarray | int) -> np.ndarray:
        """Mean-1 annual cycle evaluated at 0-based month indices."""
        m = np.asarray(month_index, dtype=float)
        phase = 2 * np.pi * (m - (self.seasonal_peak_month - 1)) / 12.0
        return 1.0 + self.seasonal_amplitude * np.cos(phase)

    def profile(self, category: str, n_days: int, start_dow: int = 0) -> np.ndarray:
        """Diurnal x weekly profile over ``n_days``, renormalised to mean 1.

        Renormalising the concatenated profile is deliberate: it keeps the
        chunk's *mean* demand equal to the base value, so volume bookkeeping
        stays exact and the weekly modulation only redistributes within the
        chunk.
        """
        d = self.diurnal(category)
        w = self.weekly(category)
        out = np.concatenate([d * w[(start_dow + k) % 7] for k in range(n_days)])
        return out / out.mean()

    def as_frame(self) -> pd.DataFrame:
        h = self._grid()
        return pd.DataFrame({"hour": h,
                             **{c: self.diurnal(c) for c in CATEGORIES}})


# ==========================================================================
# per-capita demand and capacity in population terms
# ==========================================================================

@dataclass
class DemandModelConfig:
    """Per-capita consumption, income effect, peaking, and non-residential PE."""
    litres_per_capita_day: float = 160.0
    income_factors: dict = field(default_factory=lambda: {
        "low": 0.80, "medium": 1.00, "high": 1.35})
    k1_daily_peak: float = 1.20          # max-day / mean-day
    k2_hourly_peak: float = 1.50         # max-hour / mean-hour of max day
    # non-residential intensity, expressed as population equivalents per
    # population unit assigned to that category
    pe_per_unit: dict = field(default_factory=lambda: {
        "residential": 1.0, "commercial": 1.25, "industrial": 2.0})

    def per_capita_m3s(self, income_band: str) -> float:
        f = self.income_factors.get(income_band)
        if f is None:
            raise ValueError(f"unknown income band {income_band!r}")
        return self.litres_per_capita_day * f / 1000.0 / SEC_PER_DAY


def capacity_to_population(capacity_lps: float, dm: DemandModelConfig,
                           income_band: str = "medium",
                           category_mix: dict | None = None) -> dict:
    """Convert a hydraulic capacity in L/s into a population ceiling.

    The peak factors are the point. ``capacity_lps`` from a constant-demand
    sweep is a *mean* flow the network can sustain; a real system has to survive
    the peak hour of the peak day, so the sustainable mean demand is the
    capacity divided by ``k1 * k2``. Ignoring this overstates the population a
    network can serve by roughly 80% at the default factors.
    """
    mix = category_mix or {"residential": 1.0}
    tot = sum(mix.values())
    if tot <= 0:
        raise ValueError("category_mix must have positive total weight")
    pe_weighted = sum(w / tot * dm.pe_per_unit[c] for c, w in mix.items())
    q = dm.per_capita_m3s(income_band)
    peak = dm.k1_daily_peak * dm.k2_hourly_peak
    mean_allowed_m3s = capacity_lps / LPS_PER_M3S / peak
    pop = mean_allowed_m3s / (q * pe_weighted) if q > 0 else np.nan
    return {"capacity_lps": capacity_lps,
            "peak_factor": peak,
            "sustainable_mean_lps": mean_allowed_m3s * LPS_PER_M3S,
            "population_equivalent_capacity": pop,
            "pe_weighted_intensity": pe_weighted,
            "per_capita_lps": q * LPS_PER_M3S}


# ==========================================================================
# land-use and district assignment
# ==========================================================================

@dataclass
class AssignmentConfig:
    """How junctions get land-use mixes, income bands and districts.

    Explicitly a *heuristic*, and explicitly swappable: ``classifier`` accepts a
    callable so real land-use or income layers can replace it without touching
    anything downstream. The heuristic uses only quantities a utility would
    plausibly have: served area (half the length of incident pipes), existing
    base demand, and spatial position.
    """
    n_districts: int = 5
    target_mix: dict = field(default_factory=lambda: {
        "residential": 0.70, "commercial": 0.22, "industrial": 0.08})
    mix_concentration: float = 6.0        # Dirichlet concentration around zone mix
    industrial_demand_quantile: float = 0.90   # big-demand nodes lean industrial
    income_bands: tuple[str, ...] = ("low", "medium", "high")
    income_weights: tuple[float, ...] = (0.45, 0.40, 0.15)
    seed: int = 0

    def validate(self):
        if abs(sum(self.target_mix.values()) - 1.0) > 1e-6:
            raise ValueError("target_mix must sum to 1")
        if set(self.target_mix) - set(CATEGORIES):
            raise ValueError(f"target_mix keys must be within {CATEGORIES}")
        if len(self.income_bands) != len(self.income_weights):
            raise ValueError("income_bands and income_weights must align")
        if self.n_districts < 1:
            raise ValueError("n_districts must be >= 1")
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


def assign_landuse(wn, nodes, cfg: AssignmentConfig,
                   districts: pd.Series | None = None,
                   classifier=None) -> pd.DataFrame:
    """Assign district, land-use mix, income band and initial population share.

    Zones get a mix drawn around the corpus target; junctions then get a
    Dirichlet draw around their zone's mix, so neighbours resemble each other
    without being identical. This produces spatial structure in the *shape* of
    demand, which is what a land-use drift later perturbs.

    Pass ``classifier(wn, nodes, cfg) -> DataFrame`` to substitute real data.
    """
    cfg = cfg.validate()
    if classifier is not None:
        out = classifier(wn, nodes, cfg)
        missing = {"district", "income", *CATEGORIES} - set(out.columns)
        if missing:
            raise ValueError(f"classifier output missing columns: {sorted(missing)}")
        return out

    rng = np.random.default_rng(cfg.seed)
    nodes = list(nodes)
    area = served_area_proxy(wn, nodes)
    base = pd.Series({n: sum(float(ts.base_value or 0.0) for ts in
                             wn.get_node(n).demand_timeseries_list)
                      for n in nodes}, name="base_demand_m3s")

    if districts is None:
        coords = np.array([wn.get_node(n).coordinates or (0.0, 0.0) for n in nodes],
                          dtype=float)
        if cfg.n_districts == 1 or np.allclose(coords.std(axis=0), 0):
            lab = np.zeros(len(nodes), dtype=int)
        else:
            from sklearn.cluster import KMeans
            lab = KMeans(n_clusters=cfg.n_districts, n_init=10,
                         random_state=cfg.seed).fit_predict(coords)
        districts = pd.Series(lab, index=nodes, name="district")
    else:
        districts = districts.reindex(nodes)

    # zone-level mixes: perturb the corpus target so districts differ
    target = np.array([cfg.target_mix.get(c, 0.0) for c in CATEGORIES])
    zone_mix = {}
    for d in sorted(pd.unique(districts.dropna())):
        zone_mix[d] = rng.dirichlet(np.maximum(target, 1e-3) * cfg.mix_concentration)

    ind_cut = base.quantile(cfg.industrial_demand_quantile) if len(base) else np.inf
    rows = []
    for n in nodes:
        d = districts.get(n)
        m = zone_mix.get(d, target).copy()
        # large existing demand and large served area lean industrial/commercial:
        # a proxy for the fact that big draw-offs are rarely single households
        if base.get(n, 0.0) >= ind_cut:
            m = m + np.array([0.0, 0.10, 0.25])
        w = rng.dirichlet(np.maximum(m, 1e-3) * cfg.mix_concentration)
        band = rng.choice(cfg.income_bands,
                          p=np.asarray(cfg.income_weights, dtype=float)
                          / sum(cfg.income_weights))
        rows.append({"node": n, "district": d, "income": band,
                     "served_length_m": float(area.get(n, 0.0)),
                     "base_demand_lps": float(base.get(n, 0.0)) * LPS_PER_M3S,
                     **dict(zip(CATEGORIES, w))})
    out = pd.DataFrame(rows).set_index("node")
    # initial population share proportional to served area, floored so that
    # every consumer keeps at least a token population
    share = out.served_length_m.clip(lower=out.served_length_m[
        out.served_length_m > 0].min() if (out.served_length_m > 0).any() else 1.0)
    out["pop_share"] = share / share.sum()
    return out


# ==========================================================================
# scenario: growth, land-use conversion, income shift
# ==========================================================================

@dataclass
class DriftSpec:
    """One drift primitive with explicit pre / transition / post windows.

    ``kind``:
      ``landuse_conversion`` -- move mix weight from one category to another;
        changes the **shape** of demand, identifiable from level.
      ``growth_rate``        -- multiply a district's logistic growth rate;
        changes the **level**.
      ``income_shift``       -- move households up an income band; changes the
        level only, and is therefore *not* separable from growth by design.
    """
    kind: str
    districts: tuple = ()
    start_chunk: int = 0
    transition_chunks: int = 1
    magnitude: float = 0.0
    from_category: str = "residential"
    to_category: str = "commercial"

    def validate(self, n_chunks: int):
        if self.kind not in ("landuse_conversion", "growth_rate", "income_shift"):
            raise ValueError(f"unknown drift kind {self.kind!r}")
        if self.transition_chunks < 1:
            raise ValueError("transition_chunks must be >= 1")
        if self.start_chunk < 0 or self.start_chunk >= n_chunks:
            raise ValueError("start_chunk outside the horizon")
        if self.kind == "landuse_conversion":
            for c in (self.from_category, self.to_category):
                if c not in CATEGORIES:
                    raise ValueError(f"unknown category {c!r}")
            if not 0.0 <= self.magnitude <= 1.0:
                raise ValueError("conversion magnitude must be in [0, 1]")
        return self

    def ramp(self, n_chunks: int) -> np.ndarray:
        """Smooth 0 -> 1 ramp: 0 before start, 1 after the transition window."""
        k = np.arange(n_chunks)
        x = (k - self.start_chunk) / self.transition_chunks
        return np.clip(0.5 * (1 + np.tanh(4 * (x - 0.5))), 0.0, 1.0) * \
            (k >= self.start_chunk - self.transition_chunks)

    @property
    def window(self) -> tuple[int, int]:
        return self.start_chunk, self.start_chunk + self.transition_chunks


@dataclass
class ScenarioConfig:
    """Horizon, growth, drift and uncertainty."""
    n_chunks: int = 40
    chunk_days: int = 28                 # 4 whole weeks: weekly phase stays aligned
    timestep_s: int = 3600
    start_dow: int = 0
    start_month: int = 0

    # growth
    initial_capacity_fraction: float = 0.40   # P0 = alpha * P_max
    logistic_rate_per_chunk: float = 0.045
    logistic_ceiling_fraction: float = 0.95   # of P_max
    district_rate_spread: float = 0.5         # heterogeneity across districts

    # uncertainty
    seasonal: bool = True
    noise_sigma_district: float = 0.06
    noise_sigma_common: float = 0.03
    noise_ar1: float = 0.4               # persistence of the district noise
    seed: int = 0

    # drift
    drifts: tuple = ()

    # safety
    allow_unidentifiable: bool = False

    @property
    def steps_per_chunk(self) -> int:
        return int(self.chunk_days * 24 * 3600 // self.timestep_s)

    @property
    def horizon_months(self) -> float:
        return self.n_chunks * self.chunk_days / 30.44

    def validate(self):
        if self.chunk_days % 7 != 0:
            raise ValueError(
                "chunk_days must be a multiple of 7 so the weekly profile stays "
                "phase-aligned across chunk boundaries")
        if not 0 < self.initial_capacity_fraction < 1:
            raise ValueError("initial_capacity_fraction must be in (0, 1)")
        for d in self.drifts:
            if isinstance(d, dict):
                raise TypeError(
                    "drifts contains a dict, not a DriftSpec. This happens when a "
                    "config is rebuilt from dataclasses.asdict(), which converts "
                    "nested dataclasses recursively. Use dataclasses.replace("
                    "cfg, field=value) to derive a variant instead.")
            d.validate(self.n_chunks)
        return self

    def assert_identifiable(self):
        """Reject scenarios whose ground truth cannot be recovered.

        Land use is identifiable from shape; population and income are only
        identifiable as a product. If a level drift (growth or income) overlaps
        a shape drift (land use), an observational estimator cannot attribute
        the change, and any recovery claim would be unfalsifiable. Overriding
        requires setting ``allow_unidentifiable`` deliberately.
        """
        shape = [d for d in self.drifts if d.kind == "landuse_conversion"]
        level = [d for d in self.drifts if d.kind in ("growth_rate", "income_shift")]
        clashes = []
        for a in shape:
            for b in level:
                a0, a1 = a.window
                b0, b1 = b.window
                if a0 < b1 and b0 < a1 and (set(a.districts) & set(b.districts)
                                            or not a.districts or not b.districts):
                    clashes.append((a.kind, b.kind, (a0, a1), (b0, b1)))
        if clashes and not self.allow_unidentifiable:
            raise ValueError(
                "shape drift and level drift overlap in time on shared districts, "
                f"so the ground truth is not recoverable: {clashes}. Separate the "
                "windows, or set allow_unidentifiable=True to proceed knowingly.")
        return clashes


# ==========================================================================
# trajectory construction
# ==========================================================================

def build_trajectory(assignment: pd.DataFrame, population_capacity: float,
                     cfg: ScenarioConfig, dm: DemandModelConfig) -> dict:
    """Per-chunk population, land-use mix, income factor and base demand.

    Returns tidy frames rather than a nested structure, so the ground truth is
    directly inspectable and joinable to results. Nothing here touches EPANET.
    """
    cfg = cfg.validate()
    cfg.assert_identifiable()
    rng = np.random.default_rng(cfg.seed)
    nodes = list(assignment.index)
    districts = sorted(pd.unique(assignment.district))
    K = cfg.n_chunks

    # heterogeneous growth rates so federated clients are genuinely non-IID
    rates = {d: cfg.logistic_rate_per_chunk *
             float(rng.uniform(1 - cfg.district_rate_spread,
                               1 + cfg.district_rate_spread))
             for d in districts}
    rate_mult = {d: np.ones(K) for d in districts}
    conv = {d: np.zeros(K) for d in districts}
    income_bump = {d: np.zeros(K) for d in districts}
    drift_rows = []
    for spec in cfg.drifts:
        ramp = spec.ramp(K)
        targets = list(spec.districts) if spec.districts else districts
        for d in targets:
            if spec.kind == "growth_rate":
                rate_mult[d] = rate_mult[d] * (1.0 + spec.magnitude * ramp)
            elif spec.kind == "landuse_conversion":
                conv[d] = conv[d] + spec.magnitude * ramp
            elif spec.kind == "income_shift":
                income_bump[d] = income_bump[d] + spec.magnitude * ramp
        drift_rows.append({"kind": spec.kind, "districts": str(targets),
                           "start_chunk": spec.start_chunk,
                           "transition_chunks": spec.transition_chunks,
                           "magnitude": spec.magnitude,
                           "from_category": spec.from_category,
                           "to_category": spec.to_category})

    P_max = population_capacity
    P0 = cfg.initial_capacity_fraction * P_max
    ceiling = cfg.logistic_ceiling_fraction * P_max

    # district-level logistic population
    pop_d = {}
    for d in districts:
        share = assignment.loc[assignment.district == d, "pop_share"].sum()
        p = np.zeros(K)
        p[0] = P0 * share
        for k in range(1, K):
            r = rates[d] * rate_mult[d][k]
            cap = ceiling * share
            p[k] = p[k - 1] + r * p[k - 1] * (1 - p[k - 1] / cap) if cap > 0 else 0.0
        pop_d[d] = p

    month_idx = ((np.arange(K) * cfg.chunk_days) // 30.44 + cfg.start_month) % 12
    node_rows, chunk_rows = [], []
    for k in range(K):
        seas = 1.0
        chunk_rows.append({"chunk": k, "month_index": int(month_idx[k])})
        for n in nodes:
            a = assignment.loc[n]
            d = a.district
            pop = pop_d[d][k] * (a.pop_share /
                                 max(assignment.loc[assignment.district == d,
                                                    "pop_share"].sum(), 1e-12))
            w = np.array([a[c] for c in CATEGORIES], dtype=float)
            c_from = CATEGORIES.index(cfg.drifts[0].from_category) \
                if cfg.drifts else 0
            # apply every conversion drift affecting this district
            for spec in cfg.drifts:
                if spec.kind != "landuse_conversion":
                    continue
                if spec.districts and d not in spec.districts:
                    continue
                i_from = CATEGORIES.index(spec.from_category)
                i_to = CATEGORIES.index(spec.to_category)
                moved = w[i_from] * conv[d][k]
                w = w.copy()
                w[i_from] -= moved
                w[i_to] += moved
            w = np.clip(w, 0.0, None)
            w = w / w.sum() if w.sum() > 0 else w
            band = a.income
            f_income = dm.income_factors[band] * (1.0 + income_bump[d][k])
            q = dm.litres_per_capita_day * f_income / 1000.0 / SEC_PER_DAY
            for ci, c in enumerate(CATEGORIES):
                if w[ci] <= 0:
                    continue
                base = pop * w[ci] * dm.pe_per_unit[c] * q
                node_rows.append({
                    "chunk": k, "node": n, "district": d, "category": c,
                    "population": pop, "weight": w[ci],
                    "income_factor": f_income,
                    "base_demand_m3s": base,
                    "connections": pop * w[ci] / 2.8,   # the extra observable
                })
    traj = pd.DataFrame(node_rows)
    if cfg.seasonal:
        seas_by_chunk = ShapeParams().seasonal(month_idx)
        traj["seasonal"] = traj.chunk.map(dict(enumerate(seas_by_chunk)))
        traj["base_demand_m3s"] = traj.base_demand_m3s * traj.seasonal
    else:
        traj["seasonal"] = 1.0

    return {"trajectory": traj,
            "district_population": pd.DataFrame(pop_d).rename_axis("chunk"),
            "growth_rates": pd.Series(rates, name="rate_per_chunk"),
            "drifts": pd.DataFrame(drift_rows),
            "chunks": pd.DataFrame(chunk_rows),
            "population_capacity": P_max,
            "config": asdict(cfg), "demand_model": asdict(dm)}


def district_noise(cfg: ScenarioConfig, districts, n_chunks: int,
                   steps_per_chunk: int) -> dict:
    """AR(1) multiplicative noise, correlated within a district.

    Two components: a system-wide term (weather, holidays, anything that moves
    every district at once) and a district-specific term. Independent
    per-junction noise is deliberately absent: it would average out over a
    district and so could not create the cross-client dependence the federated
    setting is about, while multiplying the pattern count by the node count.

    Each series is exponentiated and mean-corrected so its expectation is 1 and
    it introduces no volume bias.
    """
    rng = np.random.default_rng(cfg.seed + 977)
    T = n_chunks * steps_per_chunk
    phi = float(np.clip(cfg.noise_ar1, 0.0, 0.99))

    def ar1(sigma):
        e = rng.normal(0.0, sigma * np.sqrt(1 - phi ** 2), size=T)
        x = np.empty(T)
        x[0] = rng.normal(0.0, sigma)
        for t in range(1, T):
            x[t] = phi * x[t - 1] + e[t]
        return x

    common = ar1(cfg.noise_sigma_common)
    out = {}
    for d in districts:
        z = common + ar1(cfg.noise_sigma_district)
        v = np.exp(z)
        out[d] = v / v.mean()
    return out


# ==========================================================================
# capacity validation of the whole trajectory
# ==========================================================================

def validate_trajectory(traj: dict, shapes: ShapeParams, cfg: ScenarioConfig,
                        capacity_lps: float) -> pd.DataFrame:
    """Check *every* chunk's peak-hour demand against capacity before solving.

    Validating only the initial state is the trap: growth is monotone, so a
    scenario that starts comfortably can end far past the network's limit, and
    the run would waste hours before failing. Peak demand here is computed from
    the actual per-category shapes, so the check reflects the modelled rhythm
    rather than a nominal peak factor.
    """
    t = traj["trajectory"]
    n_days = cfg.chunk_days
    prof = {c: shapes.profile(c, n_days, cfg.start_dow) for c in CATEGORIES}
    rows = []
    for k, g in t.groupby("chunk"):
        total = np.zeros(len(next(iter(prof.values()))))
        for c, gc in g.groupby("category"):
            total += gc.base_demand_m3s.sum() * prof[c]
        rows.append({"chunk": int(k),
                     "mean_lps": float(total.mean()) * LPS_PER_M3S,
                     "peak_lps": float(total.max()) * LPS_PER_M3S,
                     "population": float(g.groupby("node").population.first().sum()),
                     "peak_factor_realised": float(total.max() / total.mean())})
    out = pd.DataFrame(rows)
    out["capacity_lps"] = capacity_lps
    out["headroom_ratio"] = capacity_lps / out.peak_lps
    out["breach"] = out.peak_lps > capacity_lps
    return out


# ==========================================================================
# chunked simulation
# ==========================================================================

@dataclass
class RunConfig:
    """Execution and persistence options for a long-horizon run."""
    demand_model: str = "PDD"            # PDD: unserved demand is measurable
    required_pressure_m: float = 15.0
    minimum_pressure_m: float = 0.0
    pressure_floor_m: float = 15.0
    sanity_head_m: float = 1000.0
    record_nodes: tuple | None = None    # None = every consumer
    record_links: bool = False
    carry_tank_levels: bool = True
    carry_pump_status: bool = True
    stop_on_failure: bool = False
    verbose: bool = True


def _solve(wn):
    with tempfile.TemporaryDirectory() as scratch:
        cwd = os.getcwd()
        try:
            os.chdir(scratch)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return wntr.sim.EpanetSimulator(wn).run_sim()
        finally:
            os.chdir(cwd)


def simulate_scenario(wn, traj: dict, shapes: ShapeParams, cfg: ScenarioConfig,
                      run: RunConfig | None = None) -> dict:
    """Run the horizon chunk by chunk, carrying hydraulic state forward.

    State handed across a boundary: tank levels, pump status, and the noise
    series (which is generated once for the whole horizon so it is continuous by
    construction). Weekly phase needs no handling because ``chunk_days`` is a
    multiple of 7. Controls are stateless in EPANET and are deliberately left
    unchanged as demand grows: real infrastructure lags demand, so the resulting
    degradation is a genuine regime transition rather than an artifact, and
    adapting the controls would erase the signal we want to observe.

    Per-chunk health is recorded whether or not it passes, and a failing chunk
    does not abort the run by default: a 40-chunk horizon that dies at chunk 30
    should still yield 30 usable chunks plus a diagnosis.
    """
    run = run or RunConfig()
    cfg = cfg.validate()
    t = traj["trajectory"]
    nodes = sorted(t.node.unique())
    districts = sorted(t.district.unique())
    spc = cfg.steps_per_chunk
    noise = district_noise(cfg, districts, cfg.n_chunks, spc)
    prof = {c: shapes.profile(c, cfg.chunk_days, cfg.start_dow) for c in CATEGORIES}

    w = copy.deepcopy(wn)
    # fold any file-level multiplier so our numbers mean what they say
    m0 = float(w.options.hydraulic.demand_multiplier or 1.0)
    if abs(m0 - 1.0) > 1e-12:
        for n in w.junction_name_list:
            for ts in w.get_node(n).demand_timeseries_list:
                if ts.base_value:
                    ts.base_value = float(ts.base_value) * m0
    w.options.hydraulic.demand_multiplier = 1.0
    w.options.hydraulic.demand_model = run.demand_model
    if run.demand_model.upper().startswith("P"):
        w.options.hydraulic.required_pressure = run.required_pressure_m
        w.options.hydraulic.minimum_pressure = run.minimum_pressure_m
    # EPANET reports at t = 0, ts, ..., duration, i.e. duration/ts + 1 points.
    # Using (spc - 1) steps yields exactly spc points per chunk, so no chunk
    # duplicates the first step of its pattern and the concatenated series has
    # a uniform, gap-free clock.
    w.options.time.duration = int((spc - 1) * cfg.timestep_s)
    w.options.time.hydraulic_timestep = int(cfg.timestep_s)
    w.options.time.report_timestep = int(cfg.timestep_s)
    w.options.time.pattern_timestep = int(cfg.timestep_s)

    # zero every existing demand; ours are added as explicit categories
    for n in w.junction_name_list:
        for ts in w.get_node(n).demand_timeseries_list:
            ts.base_value = 0.0
            ts.pattern_name = None

    record = list(run.record_nodes) if run.record_nodes else nodes
    tanks, pumps = list(w.tank_name_list), list(w.pump_name_list)

    press, dem, tank_lvl, pump_st, health = [], [], [], [], []
    t_start = time.time()
    for k in range(cfg.n_chunks):
        tk = t[t.chunk == k]
        # one pattern per (district, category): shape x district noise slice
        pat_names = {}
        sl = slice(k * spc, (k + 1) * spc)
        for d in districts:
            nz = noise[d][sl]
            for c in CATEGORIES:
                nm = f"_p_{d}_{c}"
                vals = prof[c][:len(nz)] * nz
                if nm in w.pattern_name_list:
                    w.remove_pattern(nm)
                w.add_pattern(nm, list(vals))
                pat_names[(d, c)] = nm

        # rebuild demand categories from the trajectory
        for n in nodes:
            node = w.get_node(n)
            node.demand_timeseries_list.clear()
        # wntr's Demands.append takes a (base, pattern, category) tuple
        for r in tk.itertuples(index=False):
            w.get_node(r.node).demand_timeseries_list.append(
                (float(r.base_demand_m3s),
                 pat_names[(r.district, r.category)],
                 r.category))
        for n in nodes:
            if len(w.get_node(n).demand_timeseries_list) == 0:
                w.get_node(n).demand_timeseries_list.append((0.0, None, "none"))

        try:
            res = _solve(w)
            ok, err = True, ""
        except Exception as exc:
            ok, err = False, f"{type(exc).__name__}: {exc}"[:200]
            health.append({"chunk": k, "converged": False, "error": err})
            if run.stop_on_failure:
                break
            continue

        cols = [c for c in record if c in res.node["pressure"].columns]
        p = res.node["pressure"][cols]
        d_act = res.node["demand"][cols]
        offset = k * spc * cfg.timestep_s
        p.index = p.index + offset
        d_act.index = d_act.index + offset
        press.append(p.astype(np.float32))
        dem.append(d_act.astype(np.float32))

        expected = 0.0
        for c, gc in tk.groupby("category"):
            expected += gc.base_demand_m3s.sum() * prof[c][:p.shape[0]].mean()
        delivered = float(d_act.to_numpy().mean(axis=0).sum())

        row = {"chunk": k, "converged": True, "error": "",
               "n_steps": int(p.shape[0]),
               "pressure_min_m": float(np.nanmin(p.to_numpy())),
               "pressure_max_m": float(np.nanmax(p.to_numpy())),
               "frac_below_floor": float((p.to_numpy() < run.pressure_floor_m).mean()),
               "frac_negative": float((p.to_numpy() < 0).mean()),
               "overpressure": bool(np.nanmax(p.to_numpy()) > run.sanity_head_m),
               "expected_lps": expected * LPS_PER_M3S,
               "delivered_lps": delivered * LPS_PER_M3S,
               "unserved_frac": float(max(0.0, 1 - delivered / expected))
               if expected > 0 else np.nan,
               "population": float(tk.groupby("node").population.first().sum())}

        if tanks:
            lv = res.node["head"][tanks] - pd.Series(
                {tt: w.get_node(tt).elevation for tt in tanks})
            lv.index = lv.index + offset
            tank_lvl.append(lv.astype(np.float32))
            ranges = {tt: (w.get_node(tt).min_level, w.get_node(tt).max_level)
                      for tt in tanks}
            at_min = np.mean([(lv[tt] <= ranges[tt][0] + 0.01).mean() for tt in tanks])
            at_max = np.mean([(lv[tt] >= ranges[tt][1] - 0.01).mean() for tt in tanks])
            row |= {"tank_frac_at_min": float(at_min),
                    "tank_frac_at_max": float(at_max),
                    "tank_net_drift_m": float(np.mean(
                        [lv[tt].iloc[-1] - lv[tt].iloc[0] for tt in tanks]))}
            if run.carry_tank_levels:
                for tt in tanks:
                    lo, hi = ranges[tt]
                    w.get_node(tt).init_level = float(
                        np.clip(lv[tt].iloc[-1], lo, hi))

        if pumps and "status" in res.link:
            st = res.link["status"][pumps]
            st.index = st.index + offset
            pump_st.append(st.astype(np.float32))
            sw = int(np.abs(np.diff(st.to_numpy(), axis=0)).sum())
            row |= {"pump_duty_cycle": float(st.to_numpy().mean()),
                    "pump_switches": sw}
            if run.carry_pump_status:
                for pn in pumps:
                    try:
                        w.get_link(pn).initial_status = (
                            "Open" if st[pn].iloc[-1] > 0.5 else "Closed")
                    except Exception:
                        pass

        health.append(row)
        if run.verbose and (k + 1) % max(1, cfg.n_chunks // 8) == 0:
            print(f"    chunk {k + 1}/{cfg.n_chunks}  "
                  f"pmin={row['pressure_min_m']:.1f}m  "
                  f"unserved={row.get('unserved_frac', float('nan')):.3f}  "
                  f"{time.time() - t_start:.0f}s")

    def cat(frames):
        return pd.concat(frames).sort_index() if frames else pd.DataFrame()

    return {"pressure": cat(press), "demand": cat(dem),
            "tank_level": cat(tank_lvl), "pump_status": cat(pump_st),
            "health": pd.DataFrame(health),
            "trajectory": traj, "config": asdict(cfg), "run": asdict(run),
            "seconds": round(time.time() - t_start, 1)}


def build_time_index(cfg: ScenarioConfig, n_steps: int) -> pd.DataFrame:
    """Calendar features for the result index: hour, day-of-week, chunk, month."""
    step_h = cfg.timestep_s / 3600.0
    t = np.arange(n_steps) * step_h
    return pd.DataFrame({
        "t_h": t,
        "hour": (t % 24).astype(int),
        "day": (t // 24).astype(int),
        "dow": ((t // 24).astype(int) + cfg.start_dow) % 7,
        "chunk": (t // (cfg.chunk_days * 24)).astype(int),
    })
