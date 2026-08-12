import math
import os
import sys
import types
import unittest
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("SECRET_KEY", "test-secret")

if "pgvector" not in sys.modules:
    pgvector_module = types.ModuleType("pgvector")
    pgvector_psycopg2_module = types.ModuleType("pgvector.psycopg2")
    pgvector_psycopg2_module.register_vector = lambda *args, **kwargs: None
    sys.modules["pgvector"] = pgvector_module
    sys.modules["pgvector.psycopg2"] = pgvector_psycopg2_module

from app.services.analytics_service import (
    _between_group_statistics,
    _descriptive_stats,
    _paired_pre_post_statistics,
)


def row(student_id, group, stage, score):
    submission = SimpleNamespace(
        id=student_id * 10 + (1 if stage == "pretest" else 2),
        attempt_number=1,
        total_score=score,
        table_json=None,
        ai_json_result=None,
    )
    task = SimpleNamespace(id=1 if stage == "pretest" else 2, assessment_stage=stage, week_number=None, mode="etalon")
    student = SimpleNamespace(id=student_id, full_name=f"Student {student_id}", experiment_group=group)
    return (submission, task, student)


class AnalyticsV2StatisticsTest(unittest.TestCase):
    def test_descriptive_stats_include_median_and_sample_sd(self):
        stats = _descriptive_stats([60, 70, 80])

        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["mean"], 70)
        self.assertEqual(stats["median"], 70)
        self.assertEqual(stats["minimum"], 60)
        self.assertEqual(stats["maximum"], 80)
        self.assertAlmostEqual(stats["standard_deviation"], 10, places=2)

    def test_paired_pre_post_statistics_use_student_level_pairs(self):
        rows = [
            row(1, "experimental", "pretest", 50),
            row(1, "experimental", "posttest", 70),
            row(2, "experimental", "pretest", 60),
            row(2, "experimental", "posttest", 72),
            row(3, "control", "pretest", 58),
        ]

        stats = _paired_pre_post_statistics(rows)

        self.assertEqual(stats["paired_count"], 2)
        self.assertEqual(stats["pretest"]["mean"], 55)
        self.assertEqual(stats["posttest"]["mean"], 71)
        self.assertEqual(stats["mean_difference"], 16)
        self.assertEqual(stats["degrees_of_freedom"], 1)
        self.assertIsNotNone(stats["cohen_dz"])
        self.assertIsNotNone(stats["p_value"])

    def test_between_group_statistics_compare_growth_scores(self):
        rows = [
            row(1, "experimental", "pretest", 40),
            row(1, "experimental", "posttest", 70),
            row(2, "experimental", "pretest", 50),
            row(2, "experimental", "posttest", 75),
            row(3, "control", "pretest", 45),
            row(3, "control", "posttest", 55),
            row(4, "control", "pretest", 50),
            row(4, "control", "posttest", 58),
        ]

        stats = _between_group_statistics(rows)

        self.assertEqual(stats["experimental_count"], 2)
        self.assertEqual(stats["control_count"], 2)
        self.assertEqual(stats["experimental_growth"], 27.5)
        self.assertEqual(stats["control_growth"], 9)
        self.assertEqual(stats["growth_difference"], 18.5)
        self.assertTrue(stats["cohen_d"] is None or math.isfinite(stats["cohen_d"]))


if __name__ == "__main__":
    unittest.main()
