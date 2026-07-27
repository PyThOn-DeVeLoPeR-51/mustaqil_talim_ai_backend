from __future__ import annotations

import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from app.services.ai_mentor_chat_service import _truncate_rag_text


class RAGExcerptTestCase(unittest.TestCase):
    def test_short_text_is_not_changed(self) -> None:
        text = "Mustaqil ta’lim talabaning mas’uliyatini oshiradi."
        self.assertEqual(_truncate_rag_text(text, 500), text)

    def test_long_text_is_cut_at_sentence_boundary(self) -> None:
        text = (
            "Birinchi gap mustaqil ta’limning ahamiyatini tushuntiradi. "
            "Ikkinchi gap metodik ta’minot haqida batafsil ma’lumot beradi. "
            "Uchinchi gap monitoring va baholash masalalarini yoritadi. "
            "To‘rtinchi gap esa keyingi mavzuga o‘tadi."
        )

        excerpt = _truncate_rag_text(text, 165)

        self.assertTrue(excerpt.endswith("…"))
        self.assertIn("beradi.", excerpt)
        self.assertNotIn("Uchinchi gap monitoring", excerpt)
        self.assertFalse(excerpt.endswith(" ma…"))

    def test_word_boundary_is_used_when_sentence_is_too_far(self) -> None:
        text = " ".join(["mustaqil"] * 100)
        excerpt = _truncate_rag_text(text, 83)

        self.assertTrue(excerpt.endswith("…"))
        self.assertNotIn("mustaq…", excerpt)
        self.assertLessEqual(len(excerpt), 84)


if __name__ == "__main__":
    unittest.main()
