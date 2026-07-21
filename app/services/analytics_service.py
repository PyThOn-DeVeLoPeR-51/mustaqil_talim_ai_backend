from collections import defaultdict
from typing import Any, cast

from sqlalchemy.orm import Session

from app.models.student import Student
from app.models.submission import Submission
from app.models.task import Task
from app.models.teacher import Teacher


PROGRESS_LABELS = [
    "Boshlang‘ich",
    "1-hafta",
    "2-hafta",
    "3-hafta",
    "4-hafta",
    "Yakuniy",
]

CRITERIA_LABELS = [
    "Proyeksiya",
    "O‘lcham",
    "Chiziqlar",
    "Aniqlik",
    "Standart",
]

PASSING_SCORE = 56.0


def _average(values: list[float]) -> float | None:
    if not values:
        return None

    return round(sum(values) / len(values), 2)


def _safe_score(value: Any) -> float | None:
    try:
        if value is None:
            return None

        score = float(value)
        return max(0.0, min(100.0, score))
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("’", "'")
        .replace("ʻ", "'")
        .replace("‘", "'")
    )


def _canonical_mode(value: str | None) -> str | None:
    normalized = _normalize_text(value)

    if normalized in {"optional", "ixtiyoriy"}:
        return "optional"

    if normalized == "etalon":
        return "etalon"

    return normalized or None


def _latest_rows_by_task(
    rows: list[tuple[Submission, Task, Student]],
) -> list[tuple[Submission, Task, Student]]:
    latest: dict[
        tuple[int, int],
        tuple[Submission, Task, Student],
    ] = {}

    for row in rows:
        submission, task, student = row
        key = (student.id, task.id)

        existing = latest.get(key)

        if existing is None:
            latest[key] = row
            continue

        existing_submission = existing[0]

        if (
            submission.attempt_number,
            submission.id,
        ) > (
            existing_submission.attempt_number,
            existing_submission.id,
        ):
            latest[key] = row

    return list(latest.values())


def _latest_rows_by_student(
    rows: list[tuple[Submission, Task, Student]],
) -> list[tuple[Submission, Task, Student]]:
    latest: dict[int, tuple[Submission, Task, Student]] = {}

    for row in rows:
        submission, _, student = row
        existing = latest.get(student.id)

        if existing is None:
            latest[student.id] = row
            continue

        existing_submission = existing[0]

        if submission.id > existing_submission.id:
            latest[student.id] = row

    return list(latest.values())


def _row_scores(
    rows: list[tuple[Submission, Task, Student]],
) -> list[float]:
    scores: list[float] = []

    for submission, _, _ in rows:
        score = _safe_score(submission.total_score)

        if score is not None:
            scores.append(score)

    return scores


def _student_average_map(
    rows: list[tuple[Submission, Task, Student]],
) -> dict[int, float]:
    scores_by_student: dict[int, list[float]] = defaultdict(list)

    for submission, _, student in rows:
        score = _safe_score(submission.total_score)

        if score is not None:
            scores_by_student[student.id].append(score)

    result: dict[int, float] = {}

    for student_id, scores in scores_by_student.items():
        average = _average(scores)

        if average is not None:
            result[student_id] = average

    return result


def _student_averages(
    rows: list[tuple[Submission, Task, Student]],
) -> list[float]:
    return list(_student_average_map(rows).values())


def _stage_student_averages(
    rows: list[tuple[Submission, Task, Student]],
    assessment_stage: str,
) -> list[float]:
    selected_rows = [
        row
        for row in rows
        if row[1].assessment_stage == assessment_stage
    ]

    return _student_averages(selected_rows)


def _week_student_averages(
    rows: list[tuple[Submission, Task, Student]],
    week_number: int,
) -> list[float]:
    selected_rows = [
        row
        for row in rows
        if row[1].week_number == week_number
    ]

    return _student_averages(selected_rows)


def _second_attempt_growth(
    rows: list[tuple[Submission, Task, Student]],
) -> float | None:
    attempts: dict[
        tuple[int, int],
        dict[int, float],
    ] = defaultdict(dict)

    for submission, task, student in rows:
        score = _safe_score(submission.total_score)

        if score is None:
            continue

        attempts[(student.id, task.id)][submission.attempt_number] = score

    differences: list[float] = []

    for attempt_scores in attempts.values():
        first_score = attempt_scores.get(1)
        second_score = attempt_scores.get(2)

        if first_score is not None and second_score is not None:
            differences.append(second_score - first_score)

    return _average(differences)


def _criterion_values_from_table(
    submission: Submission,
    aliases: tuple[str, ...],
) -> list[float]:
    values: list[float] = []

    for row in submission.table_json or []:
        if not isinstance(row, dict):
            continue

        criterion = _normalize_text(
            row.get("criterion") or row.get("Kriteriy")
        )

        if not any(alias in criterion for alias in aliases):
            continue

        score = _safe_score(row.get("score") or row.get("Ball"))

        try:
            max_score = float(
                row.get("max_score")
                or row.get("Maksimal")
                or 0
            )
        except (TypeError, ValueError):
            max_score = 0

        if score is not None and max_score > 0:
            values.append(
                max(0.0, min(100.0, score / max_score * 100))
            )

    return values


def _criterion_values_from_modules(
    submission: Submission,
    aliases: tuple[str, ...],
) -> list[float]:
    ai_result = submission.ai_json_result or {}

    if not isinstance(ai_result, dict):
        return []

    modules = ai_result.get("modules") or {}

    if not isinstance(modules, dict):
        return []

    values: list[float] = []

    for module_name, module in modules.items():
        if not isinstance(module, dict):
            continue

        normalized_name = _normalize_text(module_name)

        summary = module.get("summary") or {}
        summary_name = ""

        if isinstance(summary, dict):
            summary_name = _normalize_text(summary.get("criterion"))

        combined_name = f"{normalized_name} {summary_name}"

        if not any(alias in combined_name for alias in aliases):
            continue

        score = _safe_score(module.get("score"))

        try:
            max_score = float(module.get("max_score") or 0)
        except (TypeError, ValueError):
            max_score = 0

        if score is not None and max_score > 0:
            values.append(
                max(0.0, min(100.0, score / max_score * 100))
            )

    return values


def _criterion_score(
    submission: Submission,
    aliases: tuple[str, ...],
) -> float | None:
    values = _criterion_values_from_table(
        submission=submission,
        aliases=aliases,
    )

    if not values:
        values = _criterion_values_from_modules(
            submission=submission,
            aliases=aliases,
        )

    if values:
        return _average(values)

    return _safe_score(submission.total_score)


def _build_criteria(
    rows: list[tuple[Submission, Task, Student]],
) -> dict:
    alias_groups = [
        (
            "proyeksiya",
            "projection",
            "completeness",
            "visible_view",
        ),
        (
            "o'lcham",
            "dimension",
        ),
        (
            "chiziq",
            "line_",
            "line ",
        ),
        (
            "aniqlik",
            "tozaligi",
            "cleanliness",
            "placement",
            "joylashtirish",
        ),
        (
            "standart",
            "compliance",
            "ramka",
            "shtamp",
            "section",
            "qirqim",
        ),
    ]

    values: list[float | None] = []

    for aliases in alias_groups:
        criterion_scores: list[float] = []

        for submission, _, _ in rows:
            score = _criterion_score(submission, aliases)

            if score is not None:
                criterion_scores.append(score)

        values.append(_average(criterion_scores))

    return {
        "labels": CRITERIA_LABELS,
        "values": values,
    }


def _build_filter_options(
    db: Session,
    teacher: Teacher,
) -> dict:
    students = cast(
        list[Student],
        db.query(Student)
        .filter(Student.teacher_id == teacher.id)
        .order_by(Student.full_name.asc())
        .all(),
    )

    tasks = cast(
        list[Task],
        db.query(Task)
        .filter(Task.teacher_id == teacher.id)
        .order_by(Task.id.desc())
        .all(),
    )

    return {
        "groups": sorted({
            student.group_name
            for student in students
            if student.group_name
        }),
        "students": [
            {
                "id": student.id,
                "full_name": student.full_name,
                "group_name": student.group_name,
                "experiment_group": student.experiment_group,
            }
            for student in students
        ],
        "experiment_groups": sorted({
            student.experiment_group
            for student in students
            if student.experiment_group
            in {"experimental", "control"}
        }),
        "modes": sorted({
            canonical_mode
            for task in tasks
            if (
                canonical_mode := _canonical_mode(task.mode)
            ) in {"etalon", "optional"}
        }),
        "topics": sorted({
            task.topic
            for task in tasks
            if task.topic
        }),
        "week_numbers": sorted({
            task.week_number
            for task in tasks
            if task.week_number is not None
        }),
        "assessment_stages": sorted({
            task.assessment_stage
            for task in tasks
            if task.assessment_stage
            in {"pretest", "intermediate", "posttest"}
        }),
        "academic_periods": sorted({
            task.academic_period
            for task in tasks
            if task.academic_period
        }),
    }


def get_teacher_analytics(
    db: Session,
    teacher: Teacher,
    group_name: str | None = None,
    student_id: int | None = None,
    experiment_group: str | None = None,
    mode: str | None = None,
    topic: str | None = None,
    week_number: int | None = None,
    assessment_stage: str | None = None,
    academic_period: str | None = None,
) -> dict:
    student_query = db.query(Student).filter(
        Student.teacher_id == teacher.id
    )

    if group_name:
        student_query = student_query.filter(
            Student.group_name == group_name
        )

    if student_id is not None:
        student_query = student_query.filter(
            Student.id == student_id
        )

    if experiment_group:
        student_query = student_query.filter(
            Student.experiment_group == experiment_group
        )

    filtered_students = cast(
        list[Student],
        student_query.order_by(
            Student.full_name.asc()
        ).all(),
    )

    query = (
        db.query(Submission, Task, Student)
        .join(Task, Task.id == Submission.task_id)
        .join(Student, Student.id == Submission.student_id)
        .filter(
            Task.teacher_id == teacher.id,
            Student.teacher_id == teacher.id,
            Submission.status == "evaluated",
            Submission.total_score.isnot(None),
        )
    )

    if group_name:
        query = query.filter(Student.group_name == group_name)

    if student_id is not None:
        query = query.filter(Student.id == student_id)

    if experiment_group:
        query = query.filter(
            Student.experiment_group == experiment_group
        )

    if mode:
        canonical_mode = _canonical_mode(mode)

        if canonical_mode == "optional":
            query = query.filter(
                Task.mode.in_(["optional", "ixtiyoriy"])
            )
        else:
            query = query.filter(Task.mode == canonical_mode)

    if topic:
        query = query.filter(Task.topic == topic)

    if week_number is not None:
        query = query.filter(Task.week_number == week_number)

    if assessment_stage:
        query = query.filter(
            Task.assessment_stage == assessment_stage
        )

    if academic_period:
        query = query.filter(
            Task.academic_period == academic_period
        )

    raw_rows = query.order_by(
        Submission.created_at.asc(),
        Submission.id.asc(),
    ).all()

    rows: list[tuple[Submission, Task, Student]] = [
        (
            cast(Submission, row[0]),
            cast(Task, row[1]),
            cast(Student, row[2]),
        )
        for row in raw_rows
    ]

    latest_task_rows = _latest_rows_by_task(rows)

    initial_scores = _stage_student_averages(
        latest_task_rows,
        "pretest",
    )

    final_scores = _stage_student_averages(
        latest_task_rows,
        "posttest",
    )

    initial_average = _average(initial_scores)
    final_average = _average(final_scores)

    growth: float | None = None

    if initial_average is not None and final_average is not None:
        growth = round(final_average - initial_average, 2)

    student_average_map = _student_average_map(
        latest_task_rows
    )

    evaluated_student_count = len(student_average_map)

    success_rate: float | None = None

    if evaluated_student_count:
        passed_count = sum(
            score >= PASSING_SCORE
            for score in student_average_map.values()
        )

        success_rate = round(
            passed_count / evaluated_student_count * 100,
            2,
        )

    progress_values = [
        initial_average,
        _average(_week_student_averages(latest_task_rows, 1)),
        _average(_week_student_averages(latest_task_rows, 2)),
        _average(_week_student_averages(latest_task_rows, 3)),
        _average(_week_student_averages(latest_task_rows, 4)),
        final_average,
    ]

    group_comparison = []

    for key, label in [
        ("experimental", "Tajriba guruhi"),
        ("control", "Nazorat guruhi"),
    ]:
        group_rows = [
            row
            for row in latest_task_rows
            if row[2].experiment_group == key
        ]

        group_students = [
            student
            for student in filtered_students
            if student.experiment_group == key
        ]

        group_comparison.append({
            "label": label,
            "before": _average(
                _stage_student_averages(
                    group_rows,
                    "pretest",
                )
            ),
            "after": _average(
                _stage_student_averages(
                    group_rows,
                    "posttest",
                )
            ),
            "count": len(group_students),
        })

    distribution_counts = {
        "high": 0,
        "good": 0,
        "satisfactory": 0,
        "low": 0,
    }

    for score in student_average_map.values():
        if score >= 86:
            distribution_counts["high"] += 1
        elif score >= 71:
            distribution_counts["good"] += 1
        elif score >= 56:
            distribution_counts["satisfactory"] += 1
        else:
            distribution_counts["low"] += 1

    distribution = [
        {
            "key": "high",
            "label": "Yuqori",
            "value": distribution_counts["high"],
        },
        {
            "key": "good",
            "label": "Yaxshi",
            "value": distribution_counts["good"],
        },
        {
            "key": "satisfactory",
            "label": "Qoniqarli",
            "value": distribution_counts["satisfactory"],
        },
        {
            "key": "low",
            "label": "Past",
            "value": distribution_counts["low"],
        },
    ]

    mode_comparison = []

    for mode_key, mode_label in [
        ("etalon", "Etalon"),
        ("optional", "Ixtiyoriy"),
    ]:
        mode_rows = [
            row
            for row in latest_task_rows
            if _canonical_mode(row[1].mode) == mode_key
        ]
        mode_student_averages = _student_average_map(mode_rows)

        mode_comparison.append({
            "mode": mode_key,
            "label": mode_label,
            "average": _average(list(mode_student_averages.values())),
            "evaluated_student_count": len(mode_student_averages),
            "evaluated_result_count": len(mode_rows),
        })

    rows_by_student: dict[
        int,
        list[tuple[Submission, Task, Student]],
    ] = defaultdict(list)

    for row in latest_task_rows:
        rows_by_student[row[2].id].append(row)

    heatmap = []

    for student in filtered_students:
        student_rows = rows_by_student.get(student.id, [])

        heatmap.append({
            "student_id": student.id,
            "name": student.full_name,
            "group_name": student.group_name,
            "values": [
                _average(
                    _stage_student_averages(
                        student_rows,
                        "pretest",
                    )
                ),
                _average(
                    _week_student_averages(
                        student_rows,
                        1,
                    )
                ),
                _average(
                    _week_student_averages(
                        student_rows,
                        2,
                    )
                ),
                _average(
                    _week_student_averages(
                        student_rows,
                        3,
                    )
                ),
                _average(
                    _week_student_averages(
                        student_rows,
                        4,
                    )
                ),
                _average(
                    _stage_student_averages(
                        student_rows,
                        "posttest",
                    )
                ),
            ],
        })

    return {
        "summary": {
            "student_count": len(filtered_students),
            "evaluated_student_count": evaluated_student_count,
            "evaluated_submission_count": len(rows),
            "initial_average": initial_average,
            "final_average": final_average,
            "growth": growth,
            "second_attempt_growth": _second_attempt_growth(rows),
            "success_rate": success_rate,
        },
        "progress": {
            "labels": PROGRESS_LABELS,
            "values": progress_values,
        },
        "group_comparison": group_comparison,
        "distribution": distribution,
        "mode_comparison": mode_comparison,
        "criteria": _build_criteria(latest_task_rows),
        "heatmap": heatmap,
        "filters": _build_filter_options(
            db=db,
            teacher=teacher,
        ),
    }