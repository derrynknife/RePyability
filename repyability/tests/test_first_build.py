"""A session's first diagram does not import surpyval's joint or
degradation models to check whether its nodes are one: no model can be
one before its module is imported, and importing them (with surpyval's
regression code and pandas) took about 2 s of the first build."""

import subprocess
import sys
import textwrap


def test_a_first_diagram_imports_no_joint_or_degradation_models():
    code = textwrap.dedent("""
        import sys
        import surpyval as surv
        import repyability as r

        w = surv.Weibull.from_params([7500, 2.0])
        e = surv.Exponential.from_params([1e-4])
        edges = [("input", "p1"), ("input", "p2"), ("p1", "v"),
                 ("p2", "v"), ("v", "output")]
        rbd = r.NonRepairableRBD(edges, {"p1": w, "p2": w, "v": e})
        rbd.sf(1000.0)
        rbd.mean()
        r.RepairableRBD(
            edges,
            {n: {"reliability": w, "repairability": e} for n in ("p1", "p2")}
            | {"v": {"reliability": e, "repairability": e}},
        ).mean_availability()
        loaded = [m for m in ("surpyval.degradation",
                              "surpyval.univariate.joint") if m in sys.modules]
        print(loaded)
        """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=None,
    )
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stderr
