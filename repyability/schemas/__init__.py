"""The JSON Schemas of RePyability's documents (see ``value_plan``)."""

import json
from importlib import resources


def npv_schema() -> dict:
    """The JSON Schema (draft 2020-12) of
    [`value_plan`][repyability.value_plan]'s request and answer: its
    ``$defs`` hold ``request`` and ``answer``, each a schema of its own.

    Returns
    -------
    dict
        The schema, as ``repyability/schemas/npv-1.json`` holds it.
    """
    text = resources.files(__name__).joinpath("npv-1.json").read_text()
    return json.loads(text)
