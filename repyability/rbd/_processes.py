"""surpyval's fitted repairable-unit models as components (#269).

surpyval fits a repairable unit's failure history: a Poisson process
(``CrowAMSAA``, ``Duane``, ``HPP``) or a generalised renewal process
(``GeneralizedRenewal``, Kijima I or II). Given as a component's
``"reliability"``, each is a life and a repair RePyability already
simulates and works out:

- a Poisson process is minimal repair (Kijima ``q = 1``) of the life whose
  cumulative hazard is the process's cumulative intensity: Crow-AMSAA's
  ``(t / alpha) ** beta`` is a Weibull(``alpha``, ``beta``), Duane's
  ``b * t ** alpha`` a Weibull(``b ** (-1 / alpha)``, ``alpha``), and a
  homogeneous process's ``lambda * t`` an Exponential(``lambda``), which
  any repair renews;
- a generalised renewal process is its life distribution repaired by its
  Kijima model with its restoration factor ``q`` (0 renews, 1 is minimal,
  as a spec's ``"repair"`` takes it).

The other renewal models (ARA, ARI, G1) are refused: their repairs are not
Kijima's, and drawing their next failure needs surpyval (SurPyval#833).
"""

from typing import Any, Optional, Tuple

#: Kijima's model, by surpyval's ``kijima_type``.
_KIJIMA = {"i": "kijima1", "ii": "kijima2"}


def _translated(model) -> Optional[Tuple[Any, Optional[dict], str]]:
    """``(life, repair, name)`` for a fitted process, or None for any other
    model."""
    import surpyval as surv

    fitter = getattr(model, "dist", None)
    if fitter is surv.HPP:
        (rate,) = (float(p) for p in model.params)
        return surv.Exponential.from_params([rate]), None, "HPP"
    if fitter is surv.CrowAMSAA:
        alpha, beta = (float(p) for p in model.params)
        life = surv.Weibull.from_params([alpha, beta])
        return life, {"model": "kijima1", "q": 1.0}, "CrowAMSAA"
    if fitter is surv.Duane:
        alpha, b = (float(p) for p in model.params)
        life = surv.Weibull.from_params([b ** (-1.0 / alpha), alpha])
        return life, {"model": "kijima1", "q": 1.0}, "Duane"
    kind = str(getattr(model, "kind", ""))
    if not (hasattr(model, "restoration") and kind.endswith("Renewal")):
        return None
    kijima = _KIJIMA.get(getattr(model, "kijima_type", None))
    if kind != "Generalized Renewal" or not kijima:
        raise ValueError(
            f"its reliability is a surpyval {model.kind} model, whose "
            "repairs are not Kijima's; RePyability takes a "
            "GeneralizedRenewal (Kijima I or II), CrowAMSAA, Duane or HPP "
            "fit (drawing the others' failures needs SurPyval#833)."
        )
    repair = {"model": kijima, "q": float(model.restoration)}
    return model.model, repair, "GeneralizedRenewal"


def as_spec(name, spec):
    """``spec`` with a fitted process given as its ``"reliability"``
    written as the life and ``"repair"`` it is (see the module docstring);
    any other spec as it is."""
    if not isinstance(spec, dict):
        return spec
    try:
        translated = _translated(spec.get("reliability"))
    except ValueError as error:
        raise ValueError(f"Component {name!r}: {error}") from None
    if translated is None:
        return spec
    life, repair, kind = translated
    if "repair" in spec:
        raise ValueError(
            f"Component {name!r}: its reliability, a fitted {kind}, brings "
            "its own repair (a Poisson process is minimal repair, a "
            "generalised renewal process its Kijima model and q): give no "
            "'repair'."
        )
    out = {**spec, "reliability": life}
    if repair is not None:
        out["repair"] = repair
    return out
