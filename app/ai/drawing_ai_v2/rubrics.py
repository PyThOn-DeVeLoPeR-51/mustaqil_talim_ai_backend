"""Immutable scoring rubrics used to guard backward compatibility.

The criterion labels, order, and maximum scores are intentionally locked.  The
heuristic implementation may be refactored, but these public scoring contracts
must not drift without an explicit versioned migration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CriterionDefinition:
    key: str
    label: str
    max_score: float


ETALON_RUBRIC: tuple[CriterionDefinition, ...] = (
    CriterionDefinition("frame_titleblock", "Ramka + burchak shtampi", 3.0),
    CriterionDefinition("placement", "Chizmani to'g'ri joylashtirish", 6.0),
    CriterionDefinition("line_types", "Chiziq turlari", 8.0),
    CriterionDefinition("dimensions", "O'lcham qo'yish", 12.0),
    CriterionDefinition("projections", "Proyeksiyalar soni", 18.0),
    CriterionDefinition(
        "projection_sections",
        "Proyeksiyalarda qirqim bajarilganligi",
        10.0,
    ),
    CriterionDefinition("visible_view", "Yaqqol tasvir to'g'riligi", 24.0),
    CriterionDefinition(
        "visible_section",
        "Yaqqol tasvirda qirqim bajarilganligi",
        15.0,
    ),
    CriterionDefinition("cleanliness", "Chizma tozaligi", 4.0),
)

OPTIONAL_RUBRIC: tuple[CriterionDefinition, ...] = (
    CriterionDefinition(
        "completeness_arrangement",
        "Proyeksiyalar to‘liqligi va joylashuvi",
        15.0,
    ),
    CriterionDefinition("section_hatching", "Qirqim va shtrixovka sifati", 10.0),
    CriterionDefinition(
        "dimensions",
        "O‘lchamlar mavjudligi va joylashuvi",
        15.0,
    ),
    CriterionDefinition(
        "line_semantics",
        "Chiziq semantikasi va chizmachilik qoidalari",
        15.0,
    ),
    CriterionDefinition("cleanliness", "Chizma tozaligi va o‘qilishi", 10.0),
    CriterionDefinition("task_compliance", "Topshiriq talabiga moslik", 10.0),
)

ETALON_MAX_SCORE = sum(item.max_score for item in ETALON_RUBRIC)
OPTIONAL_RAW_MAX_SCORE = sum(item.max_score for item in OPTIONAL_RUBRIC)

assert ETALON_MAX_SCORE == 100.0
assert OPTIONAL_RAW_MAX_SCORE == 75.0
