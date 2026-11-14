"""surpyval's fitted repairable-unit models as components (#269).

surpyval fits a repairable unit's failure history: a Poisson process
(``CrowAMSAA``, ``Duane``, ``HPP``, ``CoxLewis``) or a renewal process
(``GeneralizedRenewal``, Kijima I or II; ``GeneralizedOneRenewal``;
``ARA``; ``ARI``). Given as a component's ``"reliability"``, each is a
life and a repair RePyability simulates:

- a Poisson process is minimal repair (Kijima ``q = 1``) of the life whose
  cumulative hazard is the process's cumulative intensity: Crow-AMSAA's
  ``(t / alpha) ** beta`` is a Weibull(``alpha``, ``beta``), Duane's
  ``b * t ** alpha`` a Weibull(``b ** (-1 / alpha)``, ``alpha``),
  Cox-Lewis's ``exp(alpha) * (exp(beta * t) - 1) / beta`` the life of that
  cumulative hazard (``cox_lewis_life``), and a homogeneous process's
  ``lambda * t`` an Exponential(``lambda``), which any repair renews;
- a generalised renewal process is its life distribution repaired by its
  Kijima model with its restoration factor ``q`` (0 renews, 1 is minimal,
  as a spec's ``"repair"`` takes it);
- the other renewal processes (G1, ARA, ARI) repair a unit their own way,
  which RePyability does not write out: the unit is its life from new (G1
  and ARA's lifetime distribution, ARI's baseline intensity's life) and a
  ``"repair"`` of ``{"model": "process", "process": model}``, and the
  Python simulation steps the fitted model itself, failure by failure,
  with surpyval's ``states``, ``next_failure_qf`` and ``advance``: each
  life after a repair is the model's quantile at a uniform from the
  unit's own stream, so a seeded run is reproducible. Only the
  simulations (``availability``, ``cost``, ``simulate_timelines``, by the
  Python loop) take such a unit; each event costs some tens of
  microseconds more than a plain unit's.

A generalised renewal process given as a ``"preventive"`` spec's
``"restoration"`` is its Kijima model and ``q`` alone: the task restores
the unit as the fitted repairs did. The other renewal processes are
refused there.
"""

from functools import lru_cache
from typing import Any, Optional, Tuple

#: Kijima's model, by surpyval's ``kijima_type``.
_KIJIMA = {"i": "kijima1", "ii": "kijima2"}


@lru_cache(maxsize=None)
def _poisson_process_type() -> type:
    """The class of surpyval's fitted Poisson process models (HPP,
    CrowAMSAA, Duane, CoxLewis), read off a model built with the public
    ``from_params`` since the class itself is not exported."""
    import surpyval as surv

    return type(surv.CrowAMSAA.from_params([1.0, 1.0]))


def is_poisson_process(model) -> bool:
    """Whether ``model`` is a surpyval Poisson process (``HPP``,
    ``CrowAMSAA``, ``Duane``, ``CoxLewis``, fitted or from parameters),
    whose cumulative intensity ``cif(t)`` is the expected number of
    failures by ``t``. Regressions on a Poisson process (proportional
    intensity), cause-specific processes and joint frailty models are not:
    their ``cif`` needs covariates or a cause, or leaves out an event."""
    from surpyval.recurrent import CountingProcess

    return type(model) is _poisson_process_type() and isinstance(
        getattr(model, "dist", None), CountingProcess
    )


def is_renewal_process(model) -> bool:
    """Whether ``model`` is a surpyval renewal process model
    (``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA``,
    ``ARI``), whose ``mcf`` simulates the expected number of failures."""
    from surpyval.recurrent import RenewalModel

    return isinstance(model, RenewalModel)


#: The name RePyability's Cox-Lewis life is registered under in surpyval.
COX_LEWIS_LIFE = "RePyability Cox-Lewis life"


@lru_cache(maxsize=None)
def cox_lewis_life():
    """The life whose cumulative hazard is a Cox-Lewis process's
    cumulative intensity, ``exp(alpha) * (exp(beta * t) - 1) / beta``
    (``exp(alpha) * t`` at ``beta = 0``): a surpyval
    ``CustomDistribution`` with the parameters ``alpha`` and ``beta``,
    registered once so a saved diagram with one loads. For ``beta``
    below 0 the intensity falls and the life is defective: the unit may
    never fail again."""
    import warnings

    import surpyval as surv
    from autograd import numpy as anp  # type: ignore[import-untyped]

    def Hf(x, alpha, beta):
        # expm1(b x) / b, written to hold at b = 0 (the limit x).
        small = anp.abs(beta) < 1e-12
        safe = anp.where(small, 1.0, beta)
        return anp.exp(alpha) * anp.where(small, x, anp.expm1(safe * x) / safe)

    with warnings.catch_warnings():
        # Registered once a session; a second import is not a clash.
        warnings.simplefilter("ignore", UserWarning)
        return surv.CustomDistribution(
            COX_LEWIS_LIFE,
            Hf,
            ["alpha", "beta"],
            ((None, None), (None, None)),
            (0, float("inf")),
        )


def first_life(model):
    """The life from new of a renewal process stepped by the simulation
    (G1, ARA: its lifetime distribution; ARI: the life of its baseline
    intensity)."""
    baseline = model.model
    if is_poisson_process(baseline):
        translated = _translated(baseline)
        assert translated is not None
        return translated[0]
    return baseline


def is_surpyval_model(model) -> bool:
    """Whether ``model`` is one of surpyval's own objects."""
    return type(model).__module__.partition(".")[0] == "surpyval"


def _translated(model) -> Optional[Tuple[Any, Optional[dict], str]]:
    """``(life, repair, name)`` for a fitted process, or None for any other
    model."""
    import surpyval as surv

    fitter = (
        getattr(model, "dist", None) if is_poisson_process(model) else None
    )
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
    if fitter is surv.CoxLewis:
        alpha, beta = (float(p) for p in model.params)
        life = cox_lewis_life().from_params([alpha, beta])
        return life, {"model": "kijima1", "q": 1.0}, "CoxLewis"
    if not _is_renewal(model):
        return None
    kijima = _KIJIMA.get(str(getattr(model, "kijima_type", "")))
    if str(model.kind) == "Generalized Renewal" and kijima:
        repair = {"model": kijima, "q": float(model.restoration)}
        return model.model, repair, "GeneralizedRenewal"
    return (
        first_life(model),
        {"model": "process", "process": model},
        str(model.kind),
    )


def _is_renewal(model) -> bool:
    """Whether ``model`` is a fitted surpyval renewal process."""
    return is_renewal_process(model)


def _kijima(model, what: str, takes: str) -> dict:
    """A fitted renewal process's Kijima model and restoration factor, as
    a spec's ``"repair"`` takes them; raise, saying ``what`` it is given
    as and what that ``takes``, for one whose repairs are not Kijima's."""
    kijima = _KIJIMA.get(str(getattr(model, "kijima_type", "")))
    if str(model.kind) != "Generalized Renewal" or not kijima:
        raise ValueError(
            f"{what} is a surpyval {model.kind} model, whose repairs are "
            f"not Kijima's; RePyability takes {takes}."
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
            "generalised renewal process its Kijima model and q, any other "
            "renewal process its own rule): give no 'repair'."
        )
    out = {**spec, "reliability": life}
    if repair is not None:
        out["repair"] = repair
    return out
