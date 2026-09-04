"""Controlled perturbation probe: the INTERVENTIONAL influence map.

Everything else in this project infers dependence from observed co-movement,
which is why seasonality, drift, regime mixture and self-predictability kept
confounding it. Here the cause is set by us, one node at a time, on a
completely flat network -- so the response IS the effect, with no inference.

What this measures
------------------
`d(response_j) / d(demand_i)` for every (perturbed node i, observed target j).
Because EPANET's extended-period simulation is quasi-static, a flat-demand
network sits at a single steady state and a demand perturbation moves it to
another one. There is no transport lag to find, so this is a **Jacobian, not
a dynamic response**, and `duration = 0` is the right simulation: one
hydraulic solve with tanks pinned at their initial level, which is exactly
the pure static sensitivity. Running longer would let tanks cycle and
reintroduce the very drift this design removes.

Consequences worth keeping in view
----------------------------------
* Hydraulics is nonlinear (h ~ Q^1.85), so responses do NOT superpose and
  the map is valid only near the baseline it was measured at. That is what
  the multiple perturbation levels are for -- `check_linearity` reports the
  curvature rather than assuming it away.
* Under `DD`, an over-capacity perturbation reports negative pressure
  instead of unmet demand. `min_pressure` is recorded per solve and
  infeasible runs are flagged, never silently averaged in.
* This is a DIAGNOSTIC world, not an FL dataset: flat consumption removes
  the regime signal. Use it to validate observational statistics and to
  place sensors, then go back to the drift worlds to learn.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd


# ==========================================================================
# flat baseline
# ==========================================================================
def flat_baseline(wn, anchor_scale: float = 1.0, demand_model: str = "DD",
                  min_pressure: float | None = None,
                  demand_multiplier: float = 1.0):
    """A copy of `wn` with every junction demand constant and pattern-free.

    `duration = 0` -> a single steady-state solve. Patterns are stripped
    rather than set to a flat pattern so nothing downstream can reintroduce
    a diurnal shape; tanks stay in place but act as fixed-head nodes for the
    single instant, giving the pure Jacobian.

    `demand_multiplier` MUST be pinned explicitly. `Graeme.inp` ships with
    EPANET's global multiplier at 0.2, so leaving it alone silently applies
    only a fifth of every demand AND a fifth of every perturbation -- which
    scales all sensitivities by 0.2 and, worse, measures the Jacobian at an
    operating point five times below the one the rest of the project runs
    at. `conf/base/parameters.yml` pins this to 1.0 for the same reason;
    `check_mass_balance` is what catches it if it drifts again.
    """
    w = copy.deepcopy(wn)
    w.options.hydraulic.demand_multiplier = float(demand_multiplier)
    for name in w.junction_name_list:
        j = w.get_node(name)
        base = sum(d.base_value for d in j.demand_timeseries_list)
        j.demand_timeseries_list.clear()
        j.add_demand(base=float(base) * anchor_scale, pattern_name=None)
    w.options.time.duration = 0
    w.options.hydraulic.demand_model = demand_model
    if min_pressure is not None:
        w.options.hydraulic.minimum_pressure = float(min_pressure)
    return w


def node_demands(wn) -> pd.Series:
    return pd.Series(
        {n: float(sum(d.base_value
                      for d in wn.get_node(n).demand_timeseries_list))
         for n in wn.junction_name_list})


def source_links(wn) -> list:
    """Links incident to a reservoir or tank -- where added demand must
    ultimately come from. Used by the mass-balance check."""
    sources = set(wn.reservoir_name_list) | set(wn.tank_name_list)
    out = []
    for name in wn.link_name_list:
        lk = wn.get_link(name)
        if lk.start_node_name in sources or lk.end_node_name in sources:
            sign = 1.0 if lk.start_node_name in sources else -1.0
            out.append((name, sign))
    return out


# ==========================================================================
# one solve
# ==========================================================================
def _solve(wn) -> dict:
    import wntr
    res = wntr.sim.EpanetSimulator(wn).run_sim()
    p = res.node["pressure"].iloc[0]
    q = res.link["flowrate"].iloc[0]
    junc = [n for n in wn.junction_name_list if n in p.index]
    return dict(pressure=p, flowrate=q,
                min_pressure=float(p[junc].min()),
                feasible=bool(p[junc].min() > 0))


def _set_demand(wn, node: str, value: float) -> None:
    j = wn.get_node(node)
    j.demand_timeseries_list.clear()
    j.add_demand(base=float(value), pattern_name=None)


# ==========================================================================
# the sweep
# ==========================================================================
def perturbation_sweep(wn_flat, deltas, nodes=None, verbose=False,
                       drop_infeasible: bool = True) -> tuple:
    """Perturb one node at a time; record the response everywhere.

    `deltas` are ADDITIVE increments in the network's flow units (m^3/s).
    Additive rather than multiplicative on purpose: it keeps the
    perturbation magnitude identical across nodes (so entries are
    comparable, and it gives a true partial derivative) and it still works
    for the many junctions whose base demand is zero, which a multiplier
    would leave untouched.

    Returns (long response frame, per-solve diagnostics).
    """
    base = _solve(wn_flat)
    d0 = node_demands(wn_flat)
    nodes = list(nodes) if nodes is not None else list(wn_flat.junction_name_list)

    rows, diag = [], [dict(source_node=None, delta=0.0, kind="baseline",
                           min_pressure=base["min_pressure"],
                           feasible=base["feasible"])]
    for k, i in enumerate(nodes, 1):
        for delta in deltas:
            _set_demand(wn_flat, i, d0[i] + delta)
            try:
                out = _solve(wn_flat)
            except Exception as exc:                       # noqa: BLE001
                diag.append(dict(source_node=i, delta=delta, kind="error",
                                 min_pressure=np.nan, feasible=False,
                                 note=str(exc)[:120]))
                _set_demand(wn_flat, i, d0[i])
                continue
            diag.append(dict(source_node=i, delta=delta, kind="perturbed",
                             min_pressure=out["min_pressure"],
                             feasible=out["feasible"]))
            if not (out["feasible"] or not drop_infeasible):
                _set_demand(wn_flat, i, d0[i])
                continue
            for what in ("pressure", "flowrate"):
                resp = out[what] - base[what]
                rows.append(pd.DataFrame(dict(
                    source_node=i, delta=float(delta), target=resp.index,
                    quantity=what, baseline=base[what].to_numpy(),
                    response=resp.to_numpy(),
                    sensitivity=resp.to_numpy() / float(delta))))
            _set_demand(wn_flat, i, d0[i])
        if verbose and k % 20 == 0:
            print(f"  {k}/{len(nodes)} nodes")

    sweep = (pd.concat(rows, ignore_index=True) if rows
             else pd.DataFrame(columns=["source_node", "delta", "target",
                                        "quantity", "baseline", "response",
                                        "sensitivity"]))
    return sweep, pd.DataFrame(diag)


# ==========================================================================
# physics checks -- each pinned to a quantity known by construction
# ==========================================================================
def check_baseline_flat(wn_flat, n_steps: int = 48) -> pd.DataFrame:
    """Is the 'constant' world actually constant?

    Tanks integrate net inflow, so a flat-demand network can still drift as
    they fill and drain -- which would reintroduce exactly the confound this
    design removes. Run a short extended period and measure the drift. If
    `flow_cv` is not ~0, the Jacobian at `duration=0` is still valid (tanks
    are pinned there), but any time-series reading of this world is not.
    """
    import wntr
    w = copy.deepcopy(wn_flat)
    w.options.time.duration = int(n_steps * 3600)
    w.options.time.hydraulic_timestep = 3600
    w.options.time.report_timestep = 3600
    res = wntr.sim.EpanetSimulator(w).run_sim()
    q = res.link["flowrate"]
    p = res.node["pressure"][[n for n in w.junction_name_list]]
    return pd.DataFrame([dict(
        n_tanks=len(w.tank_name_list), n_steps=len(q),
        flow_cv=float((q.std() / q.mean().abs().replace(0, np.nan))
                      .abs().median()),
        pressure_range_med=float((p.max() - p.min()).median()),
        flat=bool((q.std() / q.mean().abs().replace(0, np.nan))
                  .abs().median() < 1e-6))])


def check_mass_balance(sweep: pd.DataFrame, wn_flat) -> pd.DataFrame:
    """Added demand must reappear at the sources. Hard physical constraint --
    catches any sign or bookkeeping error in the sweep."""
    srcs = source_links(wn_flat)
    if not srcs:
        return pd.DataFrame()
    names = {n: s for n, s in srcs}
    q = sweep[(sweep["quantity"] == "flowrate")
              & (sweep["target"].isin(names))].copy()
    q["signed"] = q["response"] * q["target"].map(names)
    got = q.groupby(["source_node", "delta"])["signed"].sum().reset_index()
    got["expected"] = got["delta"]
    got["abs_error"] = (got["signed"] - got["expected"]).abs()
    got["rel_error"] = got["abs_error"] / got["delta"].abs()
    return got.rename(columns={"signed": "source_inflow_change"})


def check_diagonal_dominance(sweep: pd.DataFrame) -> pd.DataFrame:
    """Perturbing node i should move node i's own pressure most."""
    p = sweep[sweep["quantity"] == "pressure"]
    out = []
    for (i, delta), g in p.groupby(["source_node", "delta"]):
        mag = g["response"].abs()
        self_row = g[g["target"] == i]
        if not len(self_row):
            continue
        self_mag = float(mag[self_row.index[0]])
        top = float(mag.max())
        # Ties are EXPECTED and physical: perturbing a node just below the
        # source raises head loss only in the supply main, so every
        # downstream node drops by the same amount. Weak dominance within
        # 1% is the correct test -- EPANET reports pressure to 2 decimals,
        # so exact equality never survives the round-trip and strict
        # ranking would fail on a perfectly correct network.
        out.append(dict(source_node=i, delta=delta, self_mag=self_mag,
                        max_mag=top, n_targets=len(g),
                        self_is_max=bool(self_mag >= 0.99 * top),
                        self_frac_of_max=self_mag / top if top else np.nan))
    return pd.DataFrame(out)


def check_reciprocity(sweep: pd.DataFrame, wn, delta=None) -> pd.DataFrame:
    """Pressure sensitivity MUST be symmetric: dP_j/dD_i == dP_i/dD_j.

    This is Maxwell-Betti reciprocity for a resistive network, not an
    empirical accident, and it has a hard consequence for this project:
    **pressure influence can never yield direction.** Verified here at 0.1%
    relative asymmetry (EPANET report rounding) on a looped, heterogeneous
    network.

    Direction has to come from FLOW sensitivity, which is manifestly
    asymmetric -- see `supply_dependency`.
    """
    P = influence_matrix(sweep, "pressure", delta=delta)
    junc = [n for n in wn.junction_name_list
            if n in P.columns and n in P.index]
    P = P.loc[junc, junc]
    if P.empty:
        return pd.DataFrame()
    asym = (P - P.T).abs()
    mx = float(P.abs().max().max())
    worst = float(asym.max().max())
    return pd.DataFrame([dict(
        n_nodes=len(junc), max_abs_asymmetry=worst, max_abs_sensitivity=mx,
        relative_asymmetry=worst / mx if mx else np.nan,
        symmetric=bool(worst / mx < 0.01) if mx else False)])


def check_linearity(sweep: pd.DataFrame, quantity="flowrate") -> pd.DataFrame:
    """Does sensitivity hold across perturbation levels?

    Constant sensitivity => locally linear, one Jacobian describes the
    network. Drifting sensitivity is the nonlinearity, and the size of the
    drift is the honest statement of how far the map can be extrapolated.
    """
    s = sweep[sweep["quantity"] == quantity]
    piv = s.pivot_table(index=["source_node", "target"], columns="delta",
                        values="sensitivity")
    if piv.shape[1] < 2:
        return pd.DataFrame()
    lo, hi = piv.columns.min(), piv.columns.max()
    keep = piv[piv[[lo, hi]].abs().max(axis=1) > 1e-9]
    ratio = (keep[hi] / keep[lo]).replace([np.inf, -np.inf], np.nan).dropna()
    return pd.DataFrame([dict(
        quantity=quantity, delta_lo=lo, delta_hi=hi, n_pairs=len(ratio),
        ratio_median=float(ratio.median()),
        ratio_p05=float(ratio.quantile(.05)),
        ratio_p95=float(ratio.quantile(.95)),
        max_abs_dev_from_1=float((ratio - 1).abs().max()))])


# ==========================================================================
# influence maps
# ==========================================================================
def influence_matrix(sweep: pd.DataFrame, quantity="pressure", delta=None,
                     agg="mean") -> pd.DataFrame:
    """source_node x target sensitivity matrix at one perturbation level."""
    s = sweep[sweep["quantity"] == quantity]
    if delta is not None:
        s = s[s["delta"] == delta]
    return s.pivot_table(index="source_node", columns="target",
                         values="sensitivity", aggfunc=agg)


def district_influence(sweep: pd.DataFrame, districts: dict, wn,
                       quantity="pressure", delta=None, node_targets=True
                       ) -> pd.DataFrame:
    """ASYMMETRIC district x district influence.

    `M[A, B]` = mean |sensitivity| over (i in A) x (j in B).

    IMPORTANT: with `quantity="pressure"` this matrix is SYMMETRIC by
    reciprocity (see `check_reciprocity`), so it measures coupling
    STRENGTH only -- reading direction off it is a mistake. For direction
    use `supply_dependency`, which is built on flow sensitivity.
    """
    n2d = {n: d for d, nodes in districts["districts"].items() for n in nodes}
    if node_targets:
        tgt = n2d
    else:                                   # link targets -> district by ends
        tgt = {}
        for name in wn.link_name_list:
            lk = wn.get_link(name)
            da, db = n2d.get(lk.start_node_name), n2d.get(lk.end_node_name)
            if da and db and da == db:
                tgt[name] = da
    s = sweep[sweep["quantity"] == quantity].copy()
    if delta is not None:
        s = s[s["delta"] == delta]
    s["d_src"] = s["source_node"].map(n2d)
    s["d_tgt"] = s["target"].map(tgt)
    s = s.dropna(subset=["d_src", "d_tgt"])
    s["mag"] = s["sensitivity"].abs()
    return s.pivot_table(index="d_src", columns="d_tgt", values="mag",
                         aggfunc="mean")


def influence_asymmetry(M: pd.DataFrame) -> pd.DataFrame:
    """Per district pair: which direction dominates, and by how much.

    Only meaningful on a matrix that is genuinely asymmetric. On a PRESSURE
    influence matrix `asymmetry_ratio` will sit at 1.0 by reciprocity --
    that is the physics, not a bug. Use it on flow-derived matrices, or
    prefer `supply_dependency`.
    """
    ds = [d for d in M.index if d in M.columns]
    out = []
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            ab, ba = float(M.loc[a, b]), float(M.loc[b, a])
            out.append(dict(district_a=a, district_b=b,
                            infl_a_to_b=ab, infl_b_to_a=ba,
                            dominant_direction=f"{a}->{b}" if ab >= ba
                            else f"{b}->{a}",
                            asymmetry_ratio=ab / ba if ba else np.inf,
                            coupling_strength=(ab + ba) / 2))
    return pd.DataFrame(out)


def cross_district_leakage(sweep: pd.DataFrame, districts: dict,
                           quantity="pressure", delta=None) -> pd.DataFrame:
    """Within- vs between-district sensitivity -- the separation the
    observational AUROC was trying to detect, measured directly."""
    n2d = {n: d for d, nodes in districts["districts"].items() for n in nodes}
    s = sweep[sweep["quantity"] == quantity].copy()
    if delta is not None:
        s = s[s["delta"] == delta]
    s["d_src"] = s["source_node"].map(n2d)
    s["d_tgt"] = s["target"].map(n2d)
    s = s.dropna(subset=["d_src", "d_tgt"])
    s = s[s["source_node"] != s["target"]]
    s["same"] = s["d_src"] == s["d_tgt"]
    g = s.groupby("same")["sensitivity"].apply(lambda x: x.abs().mean())
    return pd.DataFrame([dict(
        within_district=float(g.get(True, np.nan)),
        between_district=float(g.get(False, np.nan)),
        ratio=float(g.get(True, np.nan) / g.get(False, np.nan))
        if g.get(False, 0) else np.inf)])


def supply_dependency(sweep: pd.DataFrame, districts: dict, wn,
                      delta=None, normalize=True, include_boundary=True
                      ) -> pd.DataFrame:
    """DIRECTED physical dependence, from flow sensitivity.

    Pressure sensitivity is symmetric (reciprocity), so direction has to
    come from flows. `dQ_p/dD_i` answers "does node i's water travel
    through pipe p", which is inherently directional: a downstream
    district's demand loads the upstream district's mains, while the
    upstream district's demand never appears in the downstream district's
    internal pipes.

    `include_boundary=True` (default) counts a district's edge pipes as
    belonging to it. Internal-only aggregation silently reports zero
    coupling for any district whose supply attaches at its own boundary
    node, because that water never enters an internal pipe -- set it False
    only to reproduce that stricter, and usually misleading, definition.

    `D[A, B]` = max over (i in B) of |dQ/dD| across pipes carried by A,
    i.e. how much of district B's demand increment transits district A.
    Read it as **"B is supplied through A"** -> A is upstream of B. With
    `normalize=True` the value is a fraction of the increment, so 1.0 means
    all of B's water crosses A and 0.0 means none does.
    """
    n2d = {n: d for d, nodes in districts["districts"].items() for n in nodes}
    internal = {}
    for name in wn.link_name_list:
        lk = wn.get_link(name)
        da, db = n2d.get(lk.start_node_name), n2d.get(lk.end_node_name)
        if da and db and da == db:
            internal.setdefault(da, []).append(name)
        elif include_boundary:
            # A pipe on the district edge belongs to BOTH its districts. Without
            # this, a district whose supply attaches at its own boundary node
            # reports zero transit for everyone -- the water never touches an
            # internal pipe. Verified on a chain built to trigger exactly that.
            for d in (da, db):
                if d:
                    internal.setdefault(d, []).append(name)

    q = sweep[sweep["quantity"] == "flowrate"].copy()
    if delta is not None:
        q = q[q["delta"] == delta]
    q["d_src"] = q["source_node"].map(n2d)
    ds = sorted(districts["districts"])
    out = pd.DataFrame(0.0, index=ds, columns=ds)
    for a in ds:                                   # transiting district
        pipes = internal.get(a, [])
        if not pipes:
            continue
        sub = q[q["target"].isin(pipes)]
        for b in ds:                               # perturbed district
            v = sub[sub["d_src"] == b]["sensitivity"].abs()
            out.loc[a, b] = float(v.max() if normalize else v.mean()) \
                if len(v) else 0.0
    out.index.name = "transits_through"            # rows = A, the upstream one
    out.columns.name = "demand_of"                 # cols = B, the perturbed one
    return out


def directed_pairs(D: pd.DataFrame) -> pd.DataFrame:
    """Ordered district pairs from `supply_dependency`, with the verdict.

    `upstream_of` is asserted only when the two directions differ by more
    than `tol`; otherwise the pair is reported as mutual/parallel supply,
    which is a real configuration in a looped network and should not be
    forced into a direction.
    """
    ds = [d for d in D.index if d in D.columns]
    out = []
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            ab, ba = float(D.loc[a, b]), float(D.loc[b, a])
            tol = 0.02 * max(abs(ab), abs(ba), 1e-12)
            if abs(ab - ba) <= tol:
                verdict = "mutual/parallel"
            else:
                verdict = f"{a} upstream of {b}" if ab > ba \
                    else f"{b} upstream of {a}"
            out.append(dict(district_a=a, district_b=b,
                            b_transits_a=ab, a_transits_b=ba,
                            verdict=verdict,
                            asymmetry=abs(ab - ba),
                            coupled=bool(max(ab, ba) > 1e-6)))
    return pd.DataFrame(out)


# ==========================================================================
# aggregation across individually-perturbed nodes
# ==========================================================================
AGGREGATORS = {
    "max": "largest response from any single perturbed node",
    "mean_all": "mean over ALL perturbed nodes (dilutes: unaffected count as 0)",
    "mean_changed": "mean over only the perturbations that moved this target",
    "median_changed": "median over perturbations that moved it (outlier-robust)",
    "sum": "total response if every node were perturbed (extensive, scales with district size)",
    "frac_changed": "fraction of perturbed nodes that moved this target at all",
    "n_changed": "count of perturbed nodes that moved this target",
}


def aggregate_response(sweep: pd.DataFrame, districts: dict, source,
                       quantity="pressure", delta=None, how="max",
                       threshold=0.0, relative_threshold=True) -> pd.Series:
    """Collapse many single-node perturbations into one value per target.

    Each perturbation is a separate experiment, so there is no single
    correct summary -- the choice changes the question being asked:

    * `max` -- worst case; "can ANY single node here move that target"
    * `mean_all` -- district-average influence, but unaffected targets pull
      it toward zero, so it conflates 'weakly coupled' with 'rarely coupled'
    * `mean_changed` -- conditional magnitude; "when it does couple, by how
      much" -- read it TOGETHER with `frac_changed`, never alone
    * `frac_changed` / `n_changed` -- breadth rather than depth
    * `sum` -- extensive, so it scales with how many nodes the district has;
      not comparable across districts of different size

    `threshold` defines "changed". With `relative_threshold` it is a
    fraction of the largest sensitivity seen anywhere in this sweep, which
    keeps it meaningful across quantities with different units.
    """
    s = sweep[sweep["quantity"] == quantity]
    if delta is not None:
        s = s[np.isclose(s["delta"], delta)]
    nodes = (districts["districts"][source]
             if source in districts["districts"] else [source])
    s = s[s["source_node"].isin(nodes)]
    if s.empty:
        return pd.Series(dtype=float)

    mag = s["sensitivity"].abs()
    thr = threshold * float(mag.max()) if relative_threshold else threshold
    s = s.assign(mag=mag, changed=mag > thr)

    g = s.groupby("target")
    if how == "max":
        out = g["mag"].max()
    elif how == "mean_all":
        out = g["mag"].mean()
    elif how == "mean_changed":
        out = s[s.changed].groupby("target")["mag"].mean()
    elif how == "median_changed":
        out = s[s.changed].groupby("target")["mag"].median()
    elif how == "sum":
        out = g["mag"].sum()
    elif how == "frac_changed":
        out = g["changed"].mean()
    elif how == "n_changed":
        out = g["changed"].sum().astype(float)
    else:
        raise ValueError(f"unknown aggregator {how!r}; "
                         f"choose from {sorted(AGGREGATORS)}")
    return out.reindex(g["mag"].max().index).fillna(0.0)


def node_positions(wn) -> dict:
    """Coordinates from the .inp, with a spring-layout fallback so the map
    still renders on a network that carries no geometry."""
    pos = {n: tuple(wn.get_node(n).coordinates)
           for n in wn.node_name_list
           if getattr(wn.get_node(n), "coordinates", None)}
    if len({p for p in pos.values()}) > 1:
        return pos
    import networkx as nx
    G = nx.Graph()
    for name in wn.link_name_list:
        lk = wn.get_link(name)
        G.add_edge(lk.start_node_name, lk.end_node_name)
    return nx.spring_layout(G, seed=0)


def plot_network_influence(wn, districts: dict, values: pd.Series,
                           source=None, quantity="pressure", how="max",
                           log=True, ax=None, cmap="inferno",
                           show_pipes=True, title=None):
    """Network map with nodes (or pipes) shaded by aggregated influence.

    The perturbed district is outlined rather than shaded, so self-influence
    -- which is always the largest and would otherwise dominate the colour
    scale -- does not hide the cross-district structure that is the point.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pos = node_positions(wn)
    n2d = {n: d for d, nodes in districts["districts"].items() for n in nodes}
    src_nodes = set(districts["districts"].get(source, []))
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 7))

    for name in wn.link_name_list:
        lk = wn.get_link(name)
        a, b = lk.start_node_name, lk.end_node_name
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    c="0.85", lw=.8, zorder=1)

    if quantity == "flowrate" and show_pipes:
        v = values[values.index.isin(wn.link_name_list)]
        vv = np.log10(v.abs() + 1e-9) if log else v.abs()
        lo, hi = float(vv.min()), float(vv.max())
        norm = plt.Normalize(lo, hi if hi > lo else lo + 1e-9)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        for name, val in vv.items():
            lk = wn.get_link(name)
            a, b = lk.start_node_name, lk.end_node_name
            if a in pos and b in pos:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        c=sm.to_rgba(val), lw=2.4, zorder=2)
        ax.scatter([pos[n][0] for n in wn.junction_name_list if n in pos],
                   [pos[n][1] for n in wn.junction_name_list if n in pos],
                   s=6, c="0.4", zorder=3)
    else:
        junc = [n for n in wn.junction_name_list
                if n in pos and n in values.index]
        v = values.reindex(junc).fillna(0.0).abs()
        vv = np.log10(v + 1e-9) if log else v
        sm = ax.scatter([pos[n][0] for n in junc], [pos[n][1] for n in junc],
                        c=vv, s=46, cmap=cmap, zorder=3,
                        edgecolors="none")
        if src_nodes:
            sel = [n for n in junc if n in src_nodes]
            ax.scatter([pos[n][0] for n in sel], [pos[n][1] for n in sel],
                       s=140, facecolors="none", edgecolors="cyan",
                       linewidths=1.6, zorder=4)

    for n in (wn.reservoir_name_list + wn.tank_name_list):
        if n in pos:
            ax.scatter(*pos[n], marker="s", s=110, c="tab:blue", zorder=5)

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"{how} |d{quantity}/dD| — perturbing {source}",
                 fontsize=10)
    handles = [Line2D([], [], marker="o", ls="", mfc="none", mec="cyan",
                      ms=10, label=f"perturbed: {source}"),
               Line2D([], [], marker="s", ls="", c="tab:blue", ms=8,
                      label="source")]
    ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=.9)
    plt.colorbar(sm, ax=ax, fraction=.04,
                 label=("log10 " if log else "") + f"|d{quantity}/dD|")
    return ax


def influence_explorer(sweep: pd.DataFrame, wn, districts: dict):
    """ipywidgets explorer: pick a perturbed district and an aggregator.

    Falls back to a static render of the first district if ipywidgets is
    unavailable, so the notebook still produces output either way.
    """
    import matplotlib.pyplot as plt
    try:
        import ipywidgets as W
        from IPython.display import display
    except ImportError:
        print("ipywidgets unavailable — static fallback")
        d0 = sorted(districts["districts"])[0]
        vals = aggregate_response(sweep, districts, d0)
        plot_network_influence(wn, districts, vals, source=d0)
        plt.show()
        return None

    ds = sorted(districts["districts"])
    deltas = sorted(sweep["delta"].unique())
    w_src = W.Dropdown(options=[(d, d) for d in ds], value=ds[0],
                       description="perturb:")
    w_q = W.Dropdown(options=[("pressure (nodes)", "pressure"),
                              ("flowrate (pipes)", "flowrate")],
                     value="pressure", description="observe:")
    w_how = W.Dropdown(options=[(f"{k} — {v}", k)
                                for k, v in AGGREGATORS.items()],
                       value="max", description="aggregate:")
    w_delta = W.Dropdown(options=[(f"{d:.6f}", d) for d in deltas],
                         value=deltas[len(deltas) // 2], description="delta:")
    w_thr = W.FloatLogSlider(value=1e-3, base=10, min=-6, max=0, step=.25,
                             description="changed >", readout_format=".1e")
    w_log = W.Checkbox(value=True, description="log colour")
    w_excl = W.Checkbox(value=True, description="hide own district")
    out = W.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            vals = aggregate_response(
                sweep, districts, w_src.value, quantity=w_q.value,
                delta=w_delta.value, how=w_how.value, threshold=w_thr.value)
            if vals.empty:
                print("no data for this combination"); return
            n2d = {n: d for d, nn in districts["districts"].items() for n in nn}
            shown = vals
            if w_excl.value and w_q.value == "pressure":
                shown = vals[[n2d.get(i) != w_src.value for i in vals.index]]
            plt.ioff()                       # VS Code renders twice otherwise
            fig, ax = plt.subplots(1, 2, figsize=(14, 5.4),
                                   gridspec_kw={"width_ratios": [2, 1]})
            plot_network_influence(wn, districts, shown, source=w_src.value,
                                   quantity=w_q.value, how=w_how.value,
                                   log=w_log.value, ax=ax[0])
            by = (pd.Series({d: vals[[n2d.get(i) == d for i in vals.index]].mean()
                             for d in ds}).fillna(0.0)
                  if w_q.value == "pressure" else pd.Series(dtype=float))
            if len(by):
                cols = ["tab:cyan" if d == w_src.value else "tab:gray"
                        for d in by.index]
                ax[1].barh(by.index, by.to_numpy(), color=cols)
                ax[1].set_xlabel(f"mean {w_how.value} |dP/dD|")
                ax[1].set_title("by district (own district highlighted)",
                                fontsize=9)
            else:
                ax[1].axis("off")
            fig.tight_layout()
            display(fig)                      # explicit; paired with close()
            plt.close(fig)
            nz = int((shown.abs() > 0).sum())
            print(f"{nz}/{len(shown)} targets affected | "
                  f"max {shown.abs().max():.4g} | "
                  f"aggregator: {AGGREGATORS[w_how.value]}")

    for w in (w_src, w_q, w_how, w_delta, w_thr, w_log, w_excl):
        w.observe(render, names="value")
    display(W.VBox([W.HBox([w_src, w_q, w_how]),
                    W.HBox([w_delta, w_thr, w_log, w_excl]), out]))
    render()
    return None      # returning `out` would make Jupyter display it a 2nd time


# ==========================================================================
# attributing observed similarity to the network
# ==========================================================================
def jacobian(sweep: pd.DataFrame, quantity="pressure", delta=None,
             targets=None) -> pd.DataFrame:
    """J[target, source_node] = d(target)/d(demand at source)."""
    M = influence_matrix(sweep, quantity=quantity, delta=delta).T
    return M.loc[targets] if targets is not None else M


def hydraulic_similarity(sweep: pd.DataFrame, districts: dict,
                         sensors: dict | None = None, quantity="pressure",
                         delta=None, demand_std: pd.Series | None = None
                         ) -> pd.DataFrame:
    """Similarity between districts implied by the NETWORK ALONE.

    The measured Jacobian says an observation is a linear mix of every
    node's demand fluctuation: ``y = J d``. Feeding it **independent**
    demands (`Cov(d)` diagonal) gives ``Cov(y) = J diag(var_d) J'`` -- the
    correlation two districts would show even with completely unrelated
    consumption, purely because they share pipes.

    That is the quantity to subtract before claiming two districts behave
    alike: anything up to this level is the plumbing, not the regime.
    """
    J = jacobian(sweep, quantity=quantity, delta=delta)
    srcs = list(J.columns)
    v = (demand_std.reindex(srcs).fillna(0.0).to_numpy() ** 2
         if demand_std is not None else np.ones(len(srcs)))

    n2d = {n: d for d, nodes in districts["districts"].items() for n in nodes}
    ds = sorted(districts["districts"])
    if sensors:
        rows = {d: [s for s in sensors.get(d, []) if s in J.index] for d in ds}
    else:
        rows = {d: [t for t in J.index if n2d.get(t) == d] for d in ds}

    sig = {}
    for d in ds:
        sig[d] = (J.loc[rows[d]].to_numpy().mean(axis=0)
                  if rows[d] else np.zeros(len(srcs)))
    out = []
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            ca = sig[a] * np.sqrt(v)
            cb = sig[b] * np.sqrt(v)
            cov = float(ca @ cb)
            den = float(np.linalg.norm(ca) * np.linalg.norm(cb))
            out.append(dict(district_a=a, district_b=b,
                            rho_hydraulic=cov / den if den else np.nan))
    return pd.DataFrame(out)


def similarity_attribution(sweep: pd.DataFrame, districts: dict,
                           observed: pd.DataFrame, sensors=None,
                           quantity="pressure", delta=None,
                           demand_std=None, obs_col="statistic"
                           ) -> pd.DataFrame:
    """How much of an OBSERVED similarity is the network, not the regime?

    `observed` needs columns district_a / district_b / `obs_col` (e.g. the
    pearson or partial_rv values from the dependence notebook).

    * `attributable_frac` = rho_hydraulic / rho_observed -- the share of the
      observed correlation reproducible with independent demands.
    * `excess` = rho_observed - rho_hydraulic -- what is left for consumption
      structure to explain. This is the number to report as "similarity
      falls by X% once hydraulic coupling is accounted for".

    Both are ratios of correlations, so they are invariant to the operating
    point that sets the Jacobian's scale. `attributable_frac > 1` means the
    pair is LESS similar than shared plumbing alone predicts -- their demand
    patterns actively oppose each other, which is a real and reportable
    finding rather than an error.
    """
    hyd = hydraulic_similarity(sweep, districts, sensors=sensors,
                               quantity=quantity, delta=delta,
                               demand_std=demand_std)
    df = observed.merge(hyd, on=["district_a", "district_b"], how="inner")
    df["rho_observed"] = df[obs_col].astype(float)
    df["attributable_frac"] = df["rho_hydraulic"] / df["rho_observed"]
    df["excess"] = df["rho_observed"] - df["rho_hydraulic"]
    df["reduction_pct"] = 100.0 * df["rho_hydraulic"] / df["rho_observed"]
    return df[["district_a", "district_b", "rho_observed", "rho_hydraulic",
               "excess", "attributable_frac", "reduction_pct"]]


# ==========================================================================
# placement that MINIMISES hydraulic confounding
# ==========================================================================
def candidate_signatures(sweep: pd.DataFrame, quantity="flowrate", delta=None,
                         demand_std: pd.Series | None = None,
                         targets=None) -> tuple:
    """Unit-norm demand-signature per candidate sensor, plus its raw norm.

    Row `c` of the Jacobian says how candidate `c` responds to every node's
    demand. Weighted by demand variability and normalised, the direction of
    that vector is *what mixture of demands the sensor reads*, and the
    cosine between two sensors' vectors IS the correlation they would show
    with independent demands -- the same quantity `hydraulic_similarity`
    reports. So placement can be optimised directly on these vectors.

    The norm is kept separately: it is the sensor's observability, and a
    near-zero norm means a dead pipe that would look beautifully decoupled
    purely because it sees nothing.
    """
    J = jacobian(sweep, quantity=quantity, delta=delta, targets=targets)
    w = (np.sqrt(demand_std.reindex(J.columns).fillna(0.0).to_numpy() ** 2)
         if demand_std is not None else np.ones(J.shape[1]))
    A = J.to_numpy() * w
    norms = np.linalg.norm(A, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    return (pd.DataFrame(A / safe[:, None], index=J.index, columns=J.columns),
            pd.Series(norms, index=J.index))


def select_decoupled(cands: pd.DataFrame, districts: dict, sigs: dict,
                     norms: dict, template: dict, w_cross=1.0, w_self=0.4,
                     min_norm_pct=25.0, seed=0) -> dict:
    """Greedy placement minimising CROSS-DISTRICT hydraulic similarity.

    The placement study maximised AUROC against adjacency, which rewards
    sensors that see shared plumbing -- `variance` wins there precisely
    because high-variance pipes are high-carry trunk pipes carrying many
    districts' water. For measuring REGIME, that is the confound, not the
    signal: the two objectives are opposed.

    Here the gain is orthogonality to the network common mode, and the
    penalty is cosine similarity with sensors already chosen in OTHER
    districts. `min_norm_pct` drops the low-observability tail so the
    optimiser cannot win by selecting pipes that see nothing.
    """
    rng = np.random.default_rng(seed)
    ds = sorted(districts["districts"])
    placement = {d: {k: [] for k in template} for d in ds}
    chosen = {k: {d: [] for d in ds} for k in template}
    taken: set = set()

    for kind, k_n in template.items():
        Sg, nm = sigs[kind], norms[kind]
        floor = np.percentile(nm[nm > 0], min_norm_pct) if (nm > 0).any() else 0
        common = Sg.to_numpy().mean(axis=0)
        cn = np.linalg.norm(common) or 1.0
        base = pd.Series(
            -np.abs(Sg.to_numpy() @ common / cn), index=Sg.index)

        pool = {d: [e for e in cands.loc[(cands.kind == kind)
                                         & (cands.district == d), "element"]
                    if e in Sg.index and nm.get(e, 0.0) >= floor]
                for d in ds}
        for slot in range(k_n):
            for d in ds:
                avail = [e for e in pool[d]
                         if e not in taken and e not in chosen[kind][d]]
                if not avail:
                    raise ValueError(f"{d}/{kind}: pool exhausted "
                                     f"(min_norm_pct={min_norm_pct} too high?)")
                other = [e for o in ds if o != d for e in chosen[kind][o]]
                own = chosen[kind][d]
                A = Sg.loc[avail].to_numpy()
                pen_x = (np.abs(A @ Sg.loc[other].to_numpy().T).max(axis=1)
                         if other else np.zeros(len(avail)))
                pen_s = (np.abs(A @ Sg.loc[own].to_numpy().T).max(axis=1)
                         if own else np.zeros(len(avail)))
                score = (base.reindex(avail).to_numpy()
                         - w_cross * pen_x - w_self * pen_s
                         + rng.normal(0, 1e-9, len(avail)))
                best = avail[int(np.argmax(score))]
                chosen[kind][d].append(best)
                taken.add(best)
                placement[d][kind].append(best)
    return placement


def compare_placements(sweep: pd.DataFrame, districts: dict,
                       placements: dict, kind="flow", quantity="flowrate",
                       delta=None, demand_std=None) -> pd.DataFrame:
    """Predicted cross-district hydraulic similarity, per placement.

    Lower is better *for regime work*: it is the share of similarity the
    plumbing would produce on its own, so a lower value leaves more room
    for consumption structure to be visible.
    """
    out = []
    for name, pl in placements.items():
        sensors = {d: list(cfg[kind]) for d, cfg in pl.items()}
        h = hydraulic_similarity(sweep, districts, sensors=sensors,
                                 quantity=quantity, delta=delta,
                                 demand_std=demand_std)
        out.append(dict(placement=name,
                        mean_rho_hydraulic=float(h.rho_hydraulic.mean()),
                        median_rho_hydraulic=float(h.rho_hydraulic.median()),
                        max_rho_hydraulic=float(h.rho_hydraulic.max()),
                        n_pairs=len(h)))
    return pd.DataFrame(out).sort_values("mean_rho_hydraulic")


# ==========================================================================
# unmixing: recover district demand from mixed sensor readings
# ==========================================================================
def district_jacobian(sweep: pd.DataFrame, districts: dict, sensors: list,
                      quantity="flowrate", delta=None,
                      node_weights: pd.Series | None = None) -> pd.DataFrame:
    """`Jd[sensor, district]` -- response to a unit demand change in a district.

    A district-level demand move distributes across its nodes roughly in
    proportion to their size, so the column is a weighted sum of node
    columns. Shape is (n_sensors x n_districts): with more sensors than
    districts the inverse problem is OVERDETERMINED, which is why aggregate
    unmixing is tractable where node-level unmixing is not.
    """
    J = jacobian(sweep, quantity=quantity, delta=delta)
    rows = [s for s in sensors if s in J.index]
    ds = sorted(districts["districts"])
    out = {}
    for d in ds:
        nodes = [n for n in districts["districts"][d] if n in J.columns]
        if not nodes:
            out[d] = np.zeros(len(rows)); continue
        w = (node_weights.reindex(nodes).fillna(0.0).to_numpy()
             if node_weights is not None else np.ones(len(nodes)))
        w = w / w.sum() if w.sum() else np.ones(len(nodes)) / len(nodes)
        out[d] = J.loc[rows, nodes].to_numpy() @ w
    return pd.DataFrame(out, index=rows)


def unmix_demands(Y: pd.DataFrame, Jd: pd.DataFrame, rcond=1e-3
                  ) -> pd.DataFrame:
    """Least-squares estimate of district demand from sensor readings.

    `Y` is (time x sensor); returns (time x district). Solves `y = Jd d`
    for `d`, which undoes the network mixing that makes `corr(y)` a poor
    estimator of `corr(d)`. Deviations from each series' own mean are used,
    so the static operating point drops out.
    """
    cols = [c for c in Jd.index if c in Y.columns]
    A = Jd.loc[cols].to_numpy()
    y = (Y[cols] - Y[cols].mean()).to_numpy()
    d = y @ np.linalg.pinv(A).T
    return pd.DataFrame(d, index=Y.index, columns=Jd.columns)


def unmixing_quality(Jd: pd.DataFrame) -> dict:
    """Conditioning of the inverse problem -- the placement objective.

    `cond` near 1 means every district is separably observable; a large
    value means some district combination is nearly invisible and its
    recovered demand will be dominated by noise. This is classical optimal
    experiment design, and it is a far better-founded placement criterion
    than either AUROC-against-adjacency or decoupling.
    """
    s = np.linalg.svd(Jd.to_numpy(), compute_uv=False)
    return dict(n_sensors=Jd.shape[0], n_districts=Jd.shape[1],
                cond=float(s.max() / s.min()) if s.min() > 0 else np.inf,
                sigma_min=float(s.min()), sigma_max=float(s.max()))


def select_dopt(cands: pd.DataFrame, districts: dict, sweep: pd.DataFrame,
                template: dict, quantity="flowrate", delta=None,
                node_weights=None, seed=0) -> dict:
    """Placement maximising the smallest singular value of the district
    Jacobian -- classical optimal experiment design.

    Unmixing error scales as 1/sigma_min, so this directly minimises the
    noise amplification of `unmix_demands`. It is the principled version of
    what `variance` achieves by accident: `variance` picks high-carry
    sensors, which happen to condition the inverse problem well.
    """
    rng = np.random.default_rng(seed)
    ds = sorted(districts["districts"])
    placement = {d: {k: [] for k in template} for d in ds}
    taken: set = set()
    for kind, k_n in template.items():
        for _ in range(k_n):
            for d in ds:
                pool = [e for e in cands.loc[(cands.kind == kind)
                                             & (cands.district == d), "element"]
                        if e not in taken]
                best, best_s = None, -np.inf
                for e in pool:
                    trial = [x for dd in ds for x in placement[dd][kind]] + [e]
                    Jd = district_jacobian(sweep, districts, trial,
                                           quantity=quantity, delta=delta,
                                           node_weights=node_weights)
                    if Jd.shape[0] < 1:
                        continue
                    s = np.linalg.svd(Jd.to_numpy(), compute_uv=False)
                    val = (float(s.min()) if len(s) >= len(ds)
                           else float(s.sum()))          # seed phase
                    val += rng.normal(0, 1e-12)
                    if val > best_s:
                        best, best_s = e, val
                if best is None:
                    raise ValueError(f"{d}/{kind}: pool exhausted")
                placement[d][kind].append(best)
                taken.add(best)
    return placement
