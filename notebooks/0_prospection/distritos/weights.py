"""The dials: how much each factor pulls nodes into the same district.

Affinity, not cost. High affinity means "keep these together"; low affinity
means "this is a good place to cut". Each factor is normalised to [0, 1] *on
this network* before mixing, so the same dial setting means the same thing on a
113-junction test network and a 6000-junction one.

Four factors, one always available
----------------------------------
* ``w_resistance`` -- pipe resistance ``L / D**5`` as baseline connection
  strength. Always available, no hydraulic solve, works on any network.
* ``w_elevation`` -- gravity systems and PRV placement both track elevation
  bands, so nodes at similar elevation likely share a pressure zone. Inert on a
  flat network; :func:`~districting.graph.check_elevation_informative` says so
  up front instead of letting the dial look broken.
* ``w_valve_cut`` -- how strongly to prefer cutting at a valve or an
  already-closed link, which are the network's own natural boundaries.
* ``w_coupling`` -- optional. A measured interventional coupling matrix, if one
  exists for this network, blended in so the cheap structural proxy can be
  checked against the real thing rather than trusted.

A weight of 0 removes its factor entirely, so the partition's sensitivity to
any one factor is read directly by zeroing the others.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd

MIX_FACTORS = ("w_resistance", "w_elevation", "w_coupling")


@dataclass
class DistrictingWeights:
    """Factor weights, on a common scale before mixing.

    ``w_resistance``, ``w_elevation`` and ``w_coupling`` are *mixing* weights:
    they are renormalised against each other, so only their ratios matter.
    ``w_valve_cut`` is different -- it is a *discount strength* applied after
    mixing, because "cut here" is not a similarity, it is a modifier on one.
    """
    w_resistance: float = 1.0
    w_elevation: float = 1.0
    w_valve_cut: float = 1.0
    w_coupling: float = 0.0                       # 0 unless a coupling map is supplied
    elevation_bandwidth_m: float | None = None    # None -> auto, from the network's own spread
    valve_cut_discount: float = 0.05
    closed_link_discount: float = 0.01

    def validate(self) -> "DistrictingWeights":
        for f in MIX_FACTORS + ("w_valve_cut",):
            if getattr(self, f) < 0:
                raise ValueError("%s must be >= 0" % f)
        for f in ("valve_cut_discount", "closed_link_discount"):
            if not 0 < getattr(self, f) <= 1:
                raise ValueError("%s must be in (0, 1]" % f)
        if sum(getattr(self, f) for f in MIX_FACTORS) <= 0:
            raise ValueError("at least one of %s must be positive" % (MIX_FACTORS,))
        return self

    def as_dict(self) -> dict:
        return asdict(self)


def composite_weights(g: nx.Graph, weights: DistrictingWeights,
                      coupling: pd.DataFrame | None = None) -> nx.Graph:
    """Return a copy of ``g`` carrying a single ``affinity`` weight per edge.

    The valve discount is applied as ``affinity * discount ** w_valve_cut``.
    That form is deliberate and is a fix, not a restyling: the previous
    ``a * (1 - w) + a * d * w`` is identical at ``w = 1`` but goes *negative*
    for ``w > 1``, which the interactive slider allowed (it ran to 2). The
    negative value was then clamped to a floor, so turning the dial past 1
    silently inverted the factor -- valve edges became the strongest edges in
    the graph and the partition started refusing to cut at valves. The
    exponential form is monotone and strictly positive for every ``w >= 0``,
    and ``w = 2`` now means what it reads as: twice as much discount.
    """
    weights.validate()
    h = g.copy()

    # --- resistance: low resistance = strongly connected = high affinity
    res = np.array([d["resistance"] for _, _, d in h.edges(data=True)])
    if res.size and res.max() > res.min():
        r_aff = 1.0 - (res - res.min()) / (res.max() - res.min())
    else:
        r_aff = np.ones(res.size)

    # --- elevation: Gaussian similarity at a bandwidth taken from the network
    el = {n: d["elevation"] for n, d in h.nodes(data=True)}
    valid_el = np.array([v for v in el.values() if np.isfinite(v)])
    bw = weights.elevation_bandwidth_m
    if bw is None:
        bw = max(float(np.std(valid_el)), 1.0) if valid_el.size else 1.0

    # --- coupling: scaled by its own 95th percentile so one near-source outlier
    #     cannot compress every other pair into the floor
    c_scale = 1.0
    if coupling is not None:
        pos = coupling.to_numpy()[coupling.to_numpy() > 0]
        c_scale = float(np.quantile(pos, 0.95)) if pos.size else 1.0

    for i, (a, b, d) in enumerate(h.edges(data=True)):
        parts, wsum = 0.0, 0.0

        if weights.w_resistance > 0:
            parts += weights.w_resistance * r_aff[i]
            wsum += weights.w_resistance

        if weights.w_elevation > 0:
            ea, eb = el.get(a, np.nan), el.get(b, np.nan)
            e_aff = (np.exp(-0.5 * ((ea - eb) / bw) ** 2)
                     if np.isfinite(ea) and np.isfinite(eb) else 0.5)
            parts += weights.w_elevation * e_aff
            wsum += weights.w_elevation

        if weights.w_coupling > 0 and coupling is not None:
            # an edge contributes only when BOTH endpoints were probed; where
            # they were not, the factor is absent rather than guessed at, and
            # the renormalisation below keeps the remaining factors on scale
            if a in coupling.index and b in coupling.columns:
                c_aff = float(np.clip(coupling.at[a, b] / max(c_scale, 1e-12), 0, 1))
                if np.isfinite(c_aff):
                    parts += weights.w_coupling * c_aff
                    wsum += weights.w_coupling

        affinity = parts / wsum if wsum > 0 else 0.5

        if weights.w_valve_cut > 0:
            if d.get("is_valve"):
                affinity *= weights.valve_cut_discount ** weights.w_valve_cut
            elif d.get("is_closed"):
                affinity *= weights.closed_link_discount ** weights.w_valve_cut

        d["affinity"] = max(float(affinity), 1e-9)
    return h
