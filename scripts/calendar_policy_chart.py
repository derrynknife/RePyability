"""Draw the user guide's chart of a calendar policy's thresholds
(docs/guide/img/calendar-thresholds.svg): the threshold age at each weekly,
quarterly and annual shut over a year of the plant in
docs/guide/maintenance.md.

The chart is drawn here, outside the package (which has no plotting
dependency). Run it again after a change to the calendar policy:

    python scripts/calendar_policy_chart.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import surpyval as surv  # noqa: E402

from repyability import Calendar, NonRepairable, Opportunity  # noqa: E402

OUT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "guide"
    / "img"
    / "calendar-thresholds.svg"
)

SERIES = (
    ("weekly", "#2a78d6", "o"),
    ("quarterly", "#eb6834", "s"),
    ("annual", "#1baf7a", "D"),
)
TEXT, MUTED, SURFACE = "#262624", "#6b6a64", "#fcfcfb"


def main() -> None:
    bearing = NonRepairable(surv.Weibull.from_params([1000.0, 2.5]))
    plant = Calendar(
        [
            Opportunity("weekly", 3.0, period=7.0),
            Opportunity("quarterly", 2.0, period=91.0),
            Opportunity("annual", 1.0, period=364.0),
        ],
        failure=20.0,
        anytime=4.0,
    )
    policy = bearing.optimal_calendar_policy(plant)
    figure, axes = plt.subplots(figsize=(8, 4.2), dpi=100)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    for name, colour, marker in SERIES:
        points = policy.schedule(kinds=(name,))
        days = [t for t, _, _, _ in points]
        ages = [a for _, _, a, _ in points]
        axes.plot(
            days,
            ages,
            linestyle="none",
            marker=marker,
            markersize=6 if name == "weekly" else 9,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=name,
            zorder=3 if name != "weekly" else 2,
        )
        if name != "weekly":
            for day, age in zip(days, ages):
                axes.annotate(
                    f"{age:.0f} d",
                    (day, age),
                    textcoords="offset points",
                    xytext=(8, -4),
                    color=TEXT,
                    fontsize=9,
                )
    axes.set_xlabel("Day of the plant's year", color=MUTED)
    axes.set_ylabel("Replace if at least this old (days)", color=MUTED)
    axes.set_xlim(-10, 374)
    axes.set_ylim(0, None)
    axes.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#c3c2b7")
    axes.tick_params(colors=MUTED)
    axes.legend(frameon=False, loc="upper left", labelcolor=TEXT)
    axes.set_title(
        "The threshold at each shut rises as the cheap annual shut nears",
        color=TEXT,
        loc="left",
        fontsize=11,
    )
    figure.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, format="svg", metadata={"Date": None})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
