"""Which houses are trustworthy enough to train on.

Two kinds of failure end up as label noise:
  - the teacher is wrong: the LiDAR pipeline mis-measured the roof (missed points, traced
    fallback outlines, small facets, things on the roof it didn't classify as building);
  - the registration is wrong: the labels were moved to the wrong place in the photo.

Each check records its value and threshold, so the Stage 2 report can show what was dropped and
why. The thresholds are PROVISIONAL: set from the four Delft houses and the LiDAR project's own
error table, to be revised from the Stage 2 distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roofphoto.geo.register import Registration

# Teacher
MIN_COVERAGE = 0.90  # share of roof points assigned to a facet (Delft houses: 0.95-0.98)
MAX_UNCERTAIN_AREA_SHARE = 0.25  # small or traced facets; their area error was up to 32% (p90)
MAX_FOOTPRINT_CHECK = 0.10  # |estimate D / report - 1| (Delft houses: 0.4-4.3%)
MAX_UNLABELLED_INTERIOR_SHARE = 0.05  # non-building points >0.5 m inside the walls, per roof point
# Registration
MIN_PIECES_PER_AXIS = 5  # inlier edge pieces constraining each house axis
MAX_RMS_M = 0.25  # residual of the better correction model (Delft houses: 0.07-0.23 m)
MAX_SHIFT_M = 1.5  # beyond the Stage 1 search; larger means a mismatch, not displacement
MAX_REPEAT_DIFF_M = 0.15  # disagreement between registrations on two JPEG renderings (~2 px)


@dataclass
class Check:
    name: str
    value: float | None
    limit: float
    passed: bool

    def as_dict(self) -> dict:
        v = None if self.value is None else round(float(self.value), 4)
        return {"value": v, "limit": self.limit, "passed": self.passed}


def _at_least(name: str, value: float | None, limit: float) -> Check:
    return Check(name, value, limit, value is not None and value >= limit)


def _at_most(name: str, value: float | None, limit: float) -> Check:
    return Check(name, value, limit, value is not None and value <= limit)


def teacher_checks(report: dict, stats: dict) -> list[Check]:
    t, q = report["totals"], report["quality"]
    area = t["roof_area_m2"]
    fc = stats.get("footprint_check_D_rel")
    return [
        _at_least("coverage", q["coverage"], MIN_COVERAGE),
        _at_most(
            "uncertain_area_share",
            t["area_in_low_confidence_facets_m2"] / area if area else None,
            MAX_UNCERTAIN_AREA_SHARE,
        ),
        _at_most("footprint_check_abs", None if fc is None else abs(fc), MAX_FOOTPRINT_CHECK),
        _at_most(
            "unlabelled_interior_share",
            stats["non_building_inside_walls"]["interior"] / max(stats["roof_points"], 1),
            MAX_UNLABELLED_INTERIOR_SHARE,
        ),
    ]


def registration_checks(reg: Registration | None) -> list[Check]:
    if reg is None:
        return [Check("registration_found", 0, 1, False)]
    support = reg.axis_support(reg.common)
    s = reg.summary()["model_comparison_same_pieces"]
    best = min(v for v in (s["rms_constant_m"], s["rms_pure_relief_m"]) if v is not None)
    return [
        _at_least("pieces_along_axis", support["along"], MIN_PIECES_PER_AXIS),
        _at_least("pieces_across_axis", support["across"], MIN_PIECES_PER_AXIS),
        _at_most("best_model_rms_m", best, MAX_RMS_M),
        _at_most("constant_shift_m", float((reg.const.s0**2).sum() ** 0.5), MAX_SHIFT_M),
    ]


def summarise(checks: list[Check]) -> dict:
    return {
        "kept": all(c.passed for c in checks),
        "failed": [c.name for c in checks if not c.passed],
        "checks": {c.name: c.as_dict() for c in checks},
    }


def repeat_diffs(a: Registration, b: Registration) -> dict[str, float]:
    """How far two registrations of the same house (on two JPEG renderings) disagree, in metres.

    Constant model: distance between the two shifts. Pure relief: difference of the two
    displacements at the house's 90th-percentile edge height, where they differ most.
    """
    h90 = float(np.percentile(np.concatenate([a.h, b.h]), 90))
    return {
        "constant_m": float(np.hypot(*(a.const.s0 - b.const.s0))),
        "relief_m": float(np.hypot(*(a.pure.v - b.pure.v)) * h90),
    }


def repeat_checks(a: Registration | None, b: Registration | None) -> list[Check]:
    """Only the constant shift decides whether a house is kept: it is the baseline label model.

    An unrepeatable pure-relief fit says that model's labels are unreliable for this house, not
    that the house is; the caller records it as a flag (see relief_reliable).
    """
    if a is None or b is None:
        return [Check("repeat_registration_found", 0, 1, False)]
    return [_at_most("repeat_diff_constant_m", repeat_diffs(a, b)["constant_m"], MAX_REPEAT_DIFF_M)]


def relief_reliable(a: Registration, b: Registration) -> bool:
    return repeat_diffs(a, b)["relief_m"] <= MAX_REPEAT_DIFF_M
