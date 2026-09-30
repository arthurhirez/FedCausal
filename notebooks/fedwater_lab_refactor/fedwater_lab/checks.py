"""One register, every check, with its verdict attached.

Each stage of the dependence workstream appends rows here instead of
printing. The point is that a claim at the end of the notebook can be traced
to the checks that were live when it was made: `Register.frame()` is a table
you can store next to the score rows, and `assert_all` is what stops a
notebook that has already failed from producing numbers.

A row is `(stage, kind, check, value, passed, note)`, and `kind` is one of
three things that must not be confused:

``check``   an INTEGRITY test -- the code, the data contract or the
            environment is wrong if it fails (the protocol contract, the
            occupancy floor, the OpenMP runtime count). `assert_all` raises
            on these.
``gate``    a VERDICT about what this world or this panel can support (is
            there active storage exchange, is the precision matrix
            interpretable, are there Markov dynamics). A closed gate is a
            result, not an error: it decides which later numbers may be read,
            and it is reported, never raised.
``report``  a measurement with no pass/fail at all (a census, a count).

`honest_n` is a row kind rather than a check: independent worlds, seeds and
pairs behind whatever was just computed, recorded next to it so that a
p-value can never be read without its sample size in the same frame.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["Register", "openmp_runtimes"]


class Register:
    """Append-only sanity register."""

    def __init__(self, label: str = ""):
        self.label = label
        self.rows: list[dict] = []

    # ------------------------------------------------------------- appending
    def add(self, stage: str, check: str, value, passed: bool | None = None,
            note: str = "", kind: str | None = None) -> "Register":
        kind = kind or ("report" if passed is None else "check")
        self.rows.append({"stage": str(stage), "kind": kind, "check": str(check),
                          "value": value, "passed": passed, "note": note})
        return self

    def report(self, stage: str, check: str, value, note: str = "") -> "Register":
        """A measurement that cannot fail (a census, a count)."""
        return self.add(stage, check, value, None, note, kind="report")

    def expect(self, stage: str, check: str, value, ok: bool,
               note: str = "") -> "Register":
        """An integrity check: `assert_all` raises when it fails."""
        return self.add(stage, check, value, bool(ok), note, kind="check")

    def gate(self, stage: str, check: str, value, open_: bool,
             note: str = "") -> "Register":
        """A verdict on what may be read. Closed is a result, not an error."""
        return self.add(stage, check, value, bool(open_), note, kind="gate")

    def honest_n(self, stage: str, worlds: int, seeds: int, pairs: int,
                 note: str = "") -> "Register":
        return self.report(stage, "honest n (worlds / seeds / pairs)",
                           f"{worlds} / {seeds} / {pairs}", note)

    def extend(self, other: "Register | pd.DataFrame", stage: str | None = None
               ) -> "Register":
        """Merge another register (or a frame of its rows) into this one."""
        rows = other.rows if isinstance(other, Register) else \
            other.to_dict("records")
        for r in rows:
            r = dict(r)
            if stage is not None:
                r["stage"] = stage
            if "kind" not in r or r["kind"] is None:
                r["kind"] = "report" if r.get("passed") is None else "check"
            self.rows.append(r)
        return self

    # -------------------------------------------------------------- reading
    def frame(self) -> pd.DataFrame:
        cols = ["stage", "kind", "check", "value", "passed", "note"]
        return pd.DataFrame(self.rows, columns=cols)

    def failed(self, stage: str | None = None) -> pd.DataFrame:
        """Integrity checks that failed. Closed gates are not failures."""
        f = self.frame()
        f = f[(f["kind"] == "check") & (f["passed"] == False)]   # noqa: E712
        return f if stage is None else f[f["stage"] == stage]

    def gates(self, open_: bool | None = None) -> pd.DataFrame:
        """Every gate verdict (optionally only the open or closed ones)."""
        f = self.frame()
        f = f[f["kind"] == "gate"]
        return f if open_ is None else f[f["passed"] == bool(open_)]

    def assert_all(self, stage: str | None = None) -> None:
        bad = self.failed(stage)
        if len(bad):
            lines = "\n".join(f"  [{r.stage}] {r.check}: {r.value}"
                              + (f"  ({r.note})" if r.note else "")
                              for r in bad.itertuples())
            raise AssertionError(
                f"{len(bad)} sanity check(s) failed"
                + (f" in stage {stage}" if stage else "") + ":\n" + lines)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        f = self.frame()
        chk, gt = f[f["kind"] == "check"], f[f["kind"] == "gate"]
        return (f"Register({self.label!r}, rows={len(f)}, "
                f"checks={len(chk)}, failed={int((chk['passed'] == False).sum())}, "  # noqa: E712
                f"gates={len(gt)}, closed={int((gt['passed'] == False).sum())})")    # noqa: E712


def openmp_runtimes() -> pd.DataFrame:
    """Every threading runtime loaded in THIS process, one row each.

    Read through `threadpoolctl` (a scikit-learn dependency). Two rows with
    `prefix == "libiomp"` is exactly the condition behind Intel's OMP Error
    #15 -- torch's own `libiomp5md.dll` plus MKL's -- which aborts the kernel
    on Windows the first time the second copy initialises. With
    `MKL_THREADING_LAYER=SEQUENTIAL` set before NumPy is imported, MKL runs
    its sequential layer, loads no OpenMP runtime, and reports
    `threading_layer == "sequential"` here.
    """
    from threadpoolctl import threadpool_info
    cols = ("user_api", "internal_api", "prefix", "threading_layer",
            "num_threads", "version", "filepath")
    return pd.DataFrame([{k: d.get(k) for k in cols} for d in threadpool_info()],
                        columns=list(cols))
