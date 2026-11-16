"""Lives on two clocks: a block whose life is a surpyval equivalent-scale
model at one usage rate.

Some parts wear by running hours and by starts. surpyval's
``Equivalence`` fits the combined clock ``hours + k starts`` and, from
surpyval 0.28, ``EquivalenceModel.at_usage(rate)`` gives the life of a
unit making ``rate`` starts per hour as a plain life on the first clock
(a ``LifeAtUsage``). That object is a block's life as it is: a
``NonRepairableRBD`` takes its ``sf``, ``ff``, ``random`` and ``mean``,
and the rate is a lever of the sensitivities (``"rate"``, see
``_model_utils.usage_spec``).

Where a diagram needs a surpyval parametric distribution (a
``NonRepairable`` and the components of a ``RepairableRBD``, whose exact
methods and engines take one), the life at the rate is that
distribution: every distribution ``Equivalence`` takes is closed under a
change of scale, and a unit used at ``rate`` ages ``a = 1 + k rate``
units of the combined clock per hour, so its life in hours is the
combined life with its time divided by ``a`` (``first_clock_life``,
``_duty.on_calendar``'s rule). The spec as given keeps the
``LifeAtUsage``, for its lever and saving, as a duty's spec keeps its
life in operating time.
"""

from typing import Any, Optional

#: What a diagram says where surpyval has no ``at_usage``.
_NEEDS = (
    "a life at a usage rate (surpyval's EquivalenceModel.at_usage) needs "
    "surpyval 0.28 or later: upgrade surpyval"
)


def life_at_usage_class() -> Optional[type]:
    """surpyval's ``LifeAtUsage``, or None for a surpyval without it
    (before 0.28)."""
    import surpyval

    return getattr(surpyval, "LifeAtUsage", None)


def has_at_usage() -> bool:
    """Whether the surpyval installed gives the life at a usage rate
    (``EquivalenceModel.at_usage``, surpyval 0.28 and later)."""
    return life_at_usage_class() is not None


def is_life_at_usage(model: Any) -> bool:
    """Whether ``model`` is surpyval's life at a usage rate
    (``EquivalenceModel.at_usage(rate)``)."""
    cls = life_at_usage_class()
    return cls is not None and isinstance(model, cls)


def _is_equivalence_model(model: Any) -> bool:
    import surpyval

    cls = getattr(surpyval, "EquivalenceModel", None)
    return cls is not None and isinstance(model, cls)


def refuse_two_clocks(model: Any) -> None:
    """Refuse a surpyval ``EquivalenceModel`` given as a block's life: it
    is a life on two clocks, and a block's life is on one, so say to give
    its life at a usage rate (or, before surpyval 0.28, which surpyval
    that needs)."""
    if not _is_equivalence_model(model):
        return
    if has_at_usage():
        raise ValueError(
            "A block's life is on one clock, and an EquivalenceModel is a "
            "life on two (its first clock plus k times its second): give "
            "the life of a unit used at one rate, model.at_usage(rate), "
            "rate the second clock's units per unit of the first (starts "
            "per hour for a model of hours and starts)."
        )
    raise ValueError(
        "A block's life is on one clock, and an EquivalenceModel is a life "
        f"on two; {_NEEDS}, then give model.at_usage(rate)."
    )


def first_clock_life(node: Any, life: Any) -> Any:
    """``life``, a ``LifeAtUsage``, as the surpyval parametric
    distribution it is on the first clock: the model's distribution with
    its time divided by ``1 + k rate``. Any other model as it is."""
    if not is_life_at_usage(life):
        return life
    from ._duty import on_calendar

    model = life.model
    combined = model.dist.from_params(list(model.dist_params))
    return on_calendar(node, combined, life.acceleration)


def to_dict(life: Any) -> dict:
    """A ``LifeAtUsage`` as JSON: its model in surpyval's format and the
    rate."""
    return {"model": life.model.to_dict(), "rate": float(life.rate)}


def from_dict(d: dict) -> Any:
    """The ``LifeAtUsage`` that ``to_dict`` saved; refused, saying what
    to do, with a surpyval that has none."""
    import surpyval

    if not has_at_usage():
        raise ValueError(f"This diagram has {_NEEDS[2:]}.")
    return surpyval.from_dict(d["model"]).at_usage(d["rate"])
