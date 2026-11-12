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

A generalised renewal process given as a ``"preventive"`` spec's
``"restoration"`` is its Kijima model and ``q`` alone: the task restores
the unit as the fitted repairs did.

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
    if not _is_renewal(model):
        return None
    repair = _kijima(
        model,
        "its reliability",
        "a GeneralizedRenewal (Kijima I or II), CrowAMSAA, Duane or HPP fit",
    )
    return model.model, repair, "GeneralizedRenewal"


def _is_renewal(model) -> bool:
    """Whether ``model`` is a fitted surpyval renewal process."""
    kind = str(getattr(model, "kind", ""))
    return hasattr(model, "restoration") and kind.endswith("Renewal")


def _kijima(model, what: str, takes: str) -> dict:
    """A fitted renewal process's Kijima model and restoration factor, as
    a spec's ``"repair"`` takes them; raise, saying ``what`` it is given
    as and what that ``takes``, for one whose repairs are not Kijima's."""
    kijima = _KIJIMA.get(str(getattr(model, "kijima_type", "")))
    if str(model.kind) != "Generalized Renewal" or not kijima:
        raise ValueError(
            f"{what} is a surpyval {model.kind} model, whose repairs are "
            f"not Kijima's; RePyability takes {takes} (drawing the "
            "others' failures needs SurPyval#833)."
        )
    return {"model": kijima, "q": float(model.restoration)}


def _as_restoration(name, preventive):
    """A ``"preventive"`` spec with a fitted generalised renewal process
    given as its ``"restoration"`` written as the Kijima model and
    restoration factor it is (as a spec's ``"repair"`` takes them); any
    other as it is."""
    if not isinstance(preventive, dict):
        return preventive
    given = preventive.get("restoration")
    if not _is_renewal(given):
        return preventive
    try:
        restoration = _kijima(
            given,
            "its preventive restoration",
            "a GeneralizedRenewal fit (Kijima I or II) or a dict of a "
            "'model' and 'q' for a restoration",
        )
    except ValueError as error:
        raise ValueError(f"Component {name!r}: {error}") from None
    return {**preventive, "restoration": restoration}


def as_spec(name, spec):
    """``spec`` with a fitted process given as its ``"reliability"``
    written as the life and ``"repair"`` it is (see the module docstring),
    and a fitted generalised renewal process given as its preventive
    ``"restoration"`` as the Kijima model and ``q`` it is; any other spec
    as it is."""
    if not isinstance(spec, dict):
        return spec
    preventive = _as_restoration(name, spec.get("preventive"))
    if preventive is not spec.get("preventive"):
        spec = {**spec, "preventive": preventive}
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
