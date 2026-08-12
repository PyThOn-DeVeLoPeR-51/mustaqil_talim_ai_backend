from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
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


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _average(values: list[float]) -> float | None:
    if not values:
        return None

    return _round(sum(values) / len(values))


def _sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (
        len(values) - 1
    )
    return _round(math.sqrt(variance))


def _descriptive_stats(values: list[float]) -> dict[str, Any]:
    cleaned = [value for value in values if math.isfinite(value)]

    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }

    return {
        "count": len(cleaned),
        "mean": _average(cleaned),
        "median": _round(float(median(cleaned))),
        "standard_deviation": _sample_sd(cleaned),
        "minimum": _round(min(cleaned)),
        "maximum": _round(max(cleaned)),
    }


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


def _percent_growth(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or abs(before) < 1e-9:
        return None

    return _round((after - before) / before * 100)


def _normal_two_sided_p_value(t_value: float) -> float:
    return max(0.0, min(1.0, math.erfc(abs(t_value) / math.sqrt(2))))


def _student_t_pdf(x: float, degrees_of_freedom: float) -> float:
    coefficient = math.exp(
        math.lgamma((degrees_of_freedom + 1) / 2)
        - math.lgamma(degrees_of_freedom / 2)
    ) / math.sqrt(degrees_of_freedom * math.pi)
    return coefficient * (
        1 + (x * x) / degrees_of_freedom
    ) ** (-(degrees_of_freedom + 1) / 2)


def _student_t_two_sided_p_value(
    t_value: float | None,
    degrees_of_freedom: float | None,
) -> float | None:
    if (
        t_value is None
        or degrees_of_freedom is None
        or degrees_of_freedom <= 0
        or not math.isfinite(t_value)
    ):
        return None

    absolute_t = abs(t_value)

    if absolute_t == 0:
        return 1.0

    if degrees_of_freedom > 120:
        return _round(_normal_two_sided_p_value(absolute_t), 4)

    # Numerical integration of the t-density from 0 to |t| with Simpson's rule.
    # This avoids adding scipy as a production dependency and is accurate enough
    # for dashboard screening. Final publication statistics should still be
    # verified with R, Jamovi, SPSS, Python/scipy, or another statistical package.
    intervals = 1200
    if intervals % 2:
        intervals += 1

    step = absolute_t / intervals
    total = _student_t_pdf(0.0, degrees_of_freedom) + _student_t_pdf(
        absolute_t,
        degrees_of_freedom,
    )

    for index in range(1, intervals):
        weight = 4 if index % 2 else 2
        total += weight * _student_t_pdf(index * step, degrees_of_freedom)

    area = total * step / 3
    cdf = 0.5 + area
    p_value = 2 * (1 - cdf)
    return _round(max(0.0, min(1.0, p_value)), 4)


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


def _stage_student_average_map(
    rows: list[tuple[Submission, Task, Student]],
    assessment_stage: str,
) -> dict[int, float]:
    selected_rows = [
        row
        for row in rows
        if row[1].assessment_stage == assessment_stage
    ]

    return _student_average_map(selected_rows)


def _stage_student_averages(
    rows: list[tuple[Submission, Task, Student]],
    assessment_stage: str,
) -> list[float]:
    return list(_stage_student_average_map(rows, assessment_stage).values())


def _week_student_average_map(
    rows: list[tuple[Submission, Task, Student]],
    week_number: int,
) -> dict[int, float]:
    selected_rows = [
        row
        for row in rows
        if row[1].week_number == week_number
    ]

    return _student_average_map(selected_rows)


def _week_student_averages(
    rows: list[tuple[Submission, Task, Student]],
    week_number: int,
) -> list[float]:
    return list(_week_student_average_map(rows, week_number).values())


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
) -> dict[str, Any]:
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


def _criterion_label(row: dict[str, Any]) -> str | None:
    raw = row.get("criterion") or row.get("Kriteriy") or row.get("name")

    if not raw:
        return None

    return str(raw).strip()


def _criterion_normalized_percent(row: dict[str, Any]) -> float | None:
    score = _safe_score(row.get("score") or row.get("Ball"))

    try:
        max_score = float(row.get("max_score") or row.get("Maksimal") or 0)
    except (TypeError, ValueError):
        max_score = 0

    if score is None or max_score <= 0:
        return None

    return max(0.0, min(100.0, score / max_score * 100))


def _build_rubric_profiles(
    rows: list[tuple[Submission, Task, Student]],
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []

    for mode_key, mode_label in [
        ("etalon", "Etalon"),
        ("optional", "Ixtiyoriy"),
    ]:
        values_by_label: dict[str, list[float]] = defaultdict(list)

        for submission, task, _ in rows:
            if _canonical_mode(task.mode) != mode_key:
                continue

            for row in submission.table_json or []:
                if not isinstance(row, dict):
                    continue

                label = _criterion_label(row)
                value = _criterion_normalized_percent(row)

                if label and value is not None:
                    values_by_label[label].append(value)

        labels = list(values_by_label.keys())
        values = [_average(values_by_label[label]) for label in labels]

        profiles.append({
            "mode": mode_key,
            "label": mode_label,
            "labels": labels,
            "values": values,
        })

    return profiles


def _paired_pre_post_rows(
    rows: list[tuple[Submission, Task, Student]],
) -> list[dict[str, Any]]:
    pretest = _stage_student_average_map(rows, "pretest")
    posttest = _stage_student_average_map(rows, "posttest")

    students_by_id = {student.id: student for _, _, student in rows}
    paired: list[dict[str, Any]] = []

    for student_id, pre_score in pretest.items():
        post_score = posttest.get(student_id)

        if post_score is None:
            continue

        student = students_by_id.get(student_id)

        paired.append({
            "student_id": student_id,
            "student": student,
            "pretest": pre_score,
            "posttest": post_score,
            "difference": post_score - pre_score,
        })

    return paired


def _paired_pre_post_statistics(
    rows: list[tuple[Submission, Task, Student]],
) -> dict[str, Any]:
    paired = _paired_pre_post_rows(rows)
    pre_scores = [row["pretest"] for row in paired]
    post_scores = [row["posttest"] for row in paired]
    differences = [row["difference"] for row in paired]

    mean_pre = _average(pre_scores)
    mean_post = _average(post_scores)
    mean_difference = _average(differences)
    sd_difference = _sample_sd(differences)
    paired_count = len(paired)

    t_value: float | None = None
    degrees_of_freedom: int | None = None
    cohen_dz: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    if paired_count >= 2 and mean_difference is not None and sd_difference:
        standard_error = sd_difference / math.sqrt(paired_count)
        t_value = mean_difference / standard_error if standard_error else None
        degrees_of_freedom = paired_count - 1
        cohen_dz = mean_difference / sd_difference
        ci_low = mean_difference - 1.96 * standard_error
        ci_high = mean_difference + 1.96 * standard_error

    p_value = _student_t_two_sided_p_value(
        t_value,
        float(degrees_of_freedom) if degrees_of_freedom is not None else None,
    )

    interpretation: str | None = None
    if mean_difference is not None:
        if mean_difference > 0:
            interpretation = "Yakuniy natija boshlang‘ich natijadan yuqori."
        elif mean_difference < 0:
            interpretation = "Yakuniy natija boshlang‘ich natijadan past."
        else:
            interpretation = "Boshlang‘ich va yakuniy natijalar teng."

    return {
        "paired_count": paired_count,
        "pretest": _descriptive_stats(pre_scores),
        "posttest": _descriptive_stats(post_scores),
        "difference": _descriptive_stats(differences),
        "mean_difference": mean_difference,
        "percent_growth": _percent_growth(mean_pre, mean_post),
        "cohen_dz": _round(cohen_dz),
        "confidence_interval_95_low": _round(ci_low),
        "confidence_interval_95_high": _round(ci_high),
        "t_value": _round(t_value),
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "interpretation": interpretation,
    }


def _group_statistics(
    rows: list[tuple[Submission, Task, Student]],
    filtered_students: list[Student],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for key, label in [
        ("experimental", "Tajriba guruhi"),
        ("control", "Nazorat guruhi"),
    ]:
        group_rows = [row for row in rows if row[2].experiment_group == key]
        paired = _paired_pre_post_rows(group_rows)
        pre_scores = [row["pretest"] for row in paired]
        post_scores = [row["posttest"] for row in paired]
        mean_pre = _average(pre_scores)
        mean_post = _average(post_scores)
        mean_growth = (
            _round(mean_post - mean_pre)
            if mean_pre is not None and mean_post is not None
            else None
        )

        result.append({
            "group": key,
            "label": label,
            "student_count": len([
                student
                for student in filtered_students
                if student.experiment_group == key
            ]),
            "paired_count": len(paired),
            "pretest": _descriptive_stats(pre_scores),
            "posttest": _descriptive_stats(post_scores),
            "mean_growth": mean_growth,
            "percent_growth": _percent_growth(mean_pre, mean_post),
        })

    return result


def _between_group_statistics(
    rows: list[tuple[Submission, Task, Student]],
) -> dict[str, Any]:
    growth_by_group: dict[str, list[float]] = {
        "experimental": [],
        "control": [],
    }

    for row in _paired_pre_post_rows(rows):
        student = row["student"]
        group = getattr(student, "experiment_group", None)

        if group in growth_by_group:
            growth_by_group[group].append(float(row["difference"]))

    experimental = growth_by_group["experimental"]
    control = growth_by_group["control"]

    exp_mean = _average(experimental)
    control_mean = _average(control)
    growth_difference = (
        _round(exp_mean - control_mean)
        if exp_mean is not None and control_mean is not None
        else None
    )

    exp_sd = _sample_sd(experimental)
    control_sd = _sample_sd(control)
    cohen_d: float | None = None
    t_value: float | None = None
    degrees_of_freedom: float | None = None

    if (
        len(experimental) >= 2
        and len(control) >= 2
        and exp_mean is not None
        and control_mean is not None
        and exp_sd is not None
        and control_sd is not None
    ):
        pooled_variance = (
            ((len(experimental) - 1) * exp_sd**2)
            + ((len(control) - 1) * control_sd**2)
        ) / (len(experimental) + len(control) - 2)

        pooled_sd = math.sqrt(pooled_variance) if pooled_variance > 0 else 0
        if pooled_sd:
            cohen_d = (exp_mean - control_mean) / pooled_sd

        standard_error_sq = (exp_sd**2 / len(experimental)) + (
            control_sd**2 / len(control)
        )

        if standard_error_sq > 0:
            t_value = (exp_mean - control_mean) / math.sqrt(standard_error_sq)
            numerator = standard_error_sq**2
            denominator = (
                (exp_sd**2 / len(experimental)) ** 2 / (len(experimental) - 1)
            ) + ((control_sd**2 / len(control)) ** 2 / (len(control) - 1))
            degrees_of_freedom = numerator / denominator if denominator else None

    p_value = _student_t_two_sided_p_value(t_value, degrees_of_freedom)

    interpretation: str | None = None
    if growth_difference is not None:
        if growth_difference > 0:
            interpretation = "Tajriba guruhi o‘sishi nazorat guruhidan yuqori."
        elif growth_difference < 0:
            interpretation = "Nazorat guruhi o‘sishi tajriba guruhidan yuqori."
        else:
            interpretation = "Tajriba va nazorat guruhlari o‘sishi teng."

    return {
        "experimental_count": len(experimental),
        "control_count": len(control),
        "experimental_growth": exp_mean,
        "control_growth": control_mean,
        "growth_difference": growth_difference,
        "cohen_d": _round(cohen_d),
        "t_value": _round(t_value),
        "degrees_of_freedom": _round(degrees_of_freedom),
        "p_value": p_value,
        "interpretation": interpretation,
    }


def _build_export_rows(
    rows: list[tuple[Submission, Task, Student]],
    filtered_students: list[Student],
) -> list[dict[str, Any]]:
    averages = _student_average_map(rows)
    pretest = _stage_student_average_map(rows, "pretest")
    posttest = _stage_student_average_map(rows, "posttest")
    weeks = {
        week_number: _week_student_average_map(rows, week_number)
        for week_number in [1, 2, 3, 4]
    }

    result: list[dict[str, Any]] = []

    for student in filtered_students:
        average = averages.get(student.id)
        growth = (
            _round(posttest[student.id] - pretest[student.id])
            if student.id in pretest and student.id in posttest
            else None
        )

        result.append({
            "student_id": student.id,
            "name": student.full_name,
            "group_name": student.group_name,
            "experiment_group": student.experiment_group,
            "pretest": pretest.get(student.id),
            "week_1": weeks[1].get(student.id),
            "week_2": weeks[2].get(student.id),
            "week_3": weeks[3].get(student.id),
            "week_4": weeks[4].get(student.id),
            "posttest": posttest.get(student.id),
            "average": average,
            "growth": growth,
            "success": average >= PASSING_SCORE if average is not None else None,
        })

    return result


def _build_filter_options(
    db: Session,
    teacher: Teacher,
) -> dict[str, Any]:
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
) -> dict[str, Any]:
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
        growth = _round(final_average - initial_average)

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

        success_rate = _round(
            passed_count / evaluated_student_count * 100,
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
        "descriptive_statistics": _descriptive_stats(
            list(student_average_map.values())
        ),
        "pre_post_statistics": _paired_pre_post_statistics(latest_task_rows),
        "group_statistics": _group_statistics(
            latest_task_rows,
            filtered_students,
        ),
        "between_group_statistics": _between_group_statistics(latest_task_rows),
        "rubric_profiles": _build_rubric_profiles(latest_task_rows),
        "export_rows": _build_export_rows(
            latest_task_rows,
            filtered_students,
        ),
        "filters": _build_filter_options(
            db=db,
            teacher=teacher,
        ),
    }
