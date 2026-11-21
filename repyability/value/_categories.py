"""The cost categories every priced result is split into."""

#: The cost categories, in the order ``CostResult.by_category`` lists them.
CATEGORIES = (
    "repair",
    "replace",
    "preventive",
    "inspection",
    "component_downtime",
    "system_downtime",
    "setup",
    # Appended: the categories' places index the engines'
    # arrays and seeded costs, so a new one goes at the end.
    "shutdown",
    # The repair crews' salaries: no engine charges them (they are the
    # same in every simulation), the tally does (``charge_crews``).
    "crews",
    # An upgrade's capital cost, at its time: worked out, not simulated.
    "capital",
)
