# %% [markdown]
# ### Mixture reconstruction — do the labelled shares rebuild the gauge?
#
# The `w_*` labels are shares of the gauge's *shape* response and drop its scale, so they cannot rebuild a
# reading. The label stack's native gains can: $a_{s,k} = \mathrm{proj}^{\mathrm{native}}_{s,k} / \mathrm{excite}^{\mathrm{native}}_k$
# is the L/s response of gauge $s$ per L/s of district $k$'s demand-profile change — for a flow link, the
# volumetric share of $k$'s demand it carries. They are applied, **not refit**, to this world's own weekly
# profiles ($Q_s$ sign-aligned gauge, $D_k$ district demand, L/s) on its `init` window and on its settled
# `final` window (the windows `verify_placement` uses).
#
# | metric | measures | definition | range · expect |
# |---|---|---|---|
# | `R2_init`, `R2_final` | do the shares rebuild the gauge's weekly shape | $\hat Q_s=\sum_k a_{s,k}D_k+c_s$, $c_s$ least squares; $R^2=1-\lVert Q_s-\hat Q_s\rVert^2/\lVert Q_s-\bar Q_s\rVert^2$ | ≤ 1 · high on both; at `init` weak evidence if district shapes are near-collinear |
# | `lam_init`, `lam_final` | do the shares account for the gauge's volume | $\lambda=\overline{\sum_k a_{s,k}D_k}\,/\,\bar Q_s$ (no $c_s$) | ≈ 1 for pure downstream demand · ≠ 1 = transit or storage flux through the link |
# | `R2_delta` | do the shares explain what the drift did to the gauge | same $R^2$ on $\Delta Q_s=Q^{\mathrm{final}}_s-Q^{\mathrm{init}}_s$ against $\sum_k a_{s,k}\Delta D_k$ | ≤ 1 · the discriminating one: only the drifting district's column is excited |
# | `R2_delta_shrunk` | robustness of `R2_delta` to noise in $a$ | `R2_delta` with $a$ from `elasticity["native"]` (projection soft-thresholded at $\nu$) | close to `R2_delta` |
# | `sign_flip` | did the link reverse between windows | aligned-profile sign differs `init` vs `final` | `False` · `True` makes every metric for that row meaningless |
#
# Pure slots are the reference: a pure gauge failing the same test means the test, not the label, is broken.

# %%
from fedwater.placement import signal_probe as sp

KIND = "flow"


def _r2(y, yhat0):
    """R² with a free intercept, per row: y, yhat0 are (n, T)."""
    e = y - yhat0
    e = e - e.mean(axis=1, keepdims=True)
    ss = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return np.where(ss > 0, 1.0 - (e ** 2).sum(axis=1) / np.where(ss > 0, ss, 1.0), np.nan)


def _windows(P, settle, patterns):
    """(init, final) step windows -- the logic of `placement.verify.verify_placement`."""
    sm, n = P.steps_month, int(P.time["n_months"])
    if float((patterns or {}).get("seasonality_scale", 0.0)) > 0:
        return mp.season_windows(P, pad=int(settle["pad"]))
    final0 = int(P.phases()["final"][0])
    try:
        s_lo, _ = mp.settled_window(P, ref_months=int(settle["ref_months"]),
                                    slack=float(settle["slack"]), pad=int(settle["pad"]),
                                    min_months=0)
        start = max(s_lo // sm, final0)
    except ValueError:
        start = final0
    need = int(settle.get("min_months", 3))
    if n - start < need:
        raise ValueError(f"{n - start} settled month(s) after the drift completes "
                         f"(month {final0}), {need} needed")
    return mp.baseline_window(P), (start * sm, n * sm)


def mixture_reconstruction(w):
    L = w.label()
    if L is None:
        print("this world uses manual sensors: no label stack"); return
    PL = w.load("sensor_placement")
    PL = PL[PL["kind"] == KIND].reset_index(drop=True)
    sch = w.load("gt_drift_schedule")
    if sch is None or not len(sch):
        print("no drift in this world: nothing to reconstruct against"); return
    tgt = str(sch.sort_values("drift_month")["district"].iloc[0])
    dy = w.load("districts")
    patterns = w.params["patterns"]
    P = sp.Probe(districts=dy["districts"], assets=dy.get("assets") or {},
                 demand=w.load("demand_series"),
                 pressures=pd.DataFrame(),           # flow only: pressures never read
                 flows=w.load("flows"), schedule=sch, time=w.params["time"],
                 params={"patterns": patterns}, drift={"tgt_district": tgt}, label="world")

    # a from the LABEL stack: the stack the placement's w_* came from
    G = L.gains[L.gains["kind"] == KIND]
    A = (G.assign(a=G["proj_native"] / G["excite_native"])
          .pivot(index="id", columns="drift_district", values="a"))
    assert set(A.columns) == set(P.names), (sorted(A.columns), P.names)
    A = A.reindex(index=PL["sensor"], columns=P.names)
    A_s = L.elasticity["native"].reindex(index=PL["sensor"], columns=P.names)
    W = [f"w_{d}" for d in P.names]
    lw = L.mixture.set_index("id")[W].reindex(PL["sensor"]).to_numpy()
    print(f"placement w_* vs label-stack mixture, max |diff|: "
          f"{np.nanmax(np.abs(lw - PL[W].to_numpy())):.2e}")

    try:
        (i0, i1), (f0, f1) = _windows(P, w.params["sensor_placement"]["probe"]["settle"], patterns)
    except ValueError as exc:
        print(f"unsettled: {exc}"); return
    cand = pd.DataFrame({"id": PL["sensor"].to_numpy(), "kind": KIND,
                         "district": PL["district"].to_numpy(), "role": "",
                         "element": PL["element"].astype(str).to_numpy()})
    Qi, sgn_i = sp.aligned_profile(P, cand, i0, i1)
    Qf, sgn_f = sp.aligned_profile(P, cand, f0, f1)
    Di, Df = sp.district_profile(P, i0, i1), sp.district_profile(P, f0, f1)

    a, a_s = A.to_numpy(), A_s.to_numpy()
    Hi, Hf = a @ Di, a @ Df
    out = PL[["district", "slot_class", "slot", "sensor", "tier", "purity"]].copy()
    out["R2_init"], out["R2_final"] = _r2(Qi, Hi), _r2(Qf, Hf)
    out["lam_init"] = Hi.mean(axis=1) / Qi.mean(axis=1)
    out["lam_final"] = Hf.mean(axis=1) / Qf.mean(axis=1)
    out["R2_delta"] = _r2(Qf - Qi, Hf - Hi)
    out["R2_delta_shrunk"] = _r2(Qf - Qi, a_s @ (Df - Di))
    out["sign_flip"] = sgn_i != sgn_f
    out["a_missing"] = np.isnan(a).any(axis=1)

    print(f"drift district: {tgt} | init steps {i0}..{i1} | final steps {f0}..{f1} "
          f"(months {f0 // P.steps_month}..{f1 // P.steps_month})")
    display(out.sort_values(["district", "slot_class", "slot"]).round(3).reset_index(drop=True))


SEL.show(mixture_reconstruction);
