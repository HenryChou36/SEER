from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from seer.evaluate import resumed_rows


class EvaluationProgressTest(unittest.TestCase):
    @staticmethod
    def sources(count: int) -> list[dict[str, str]]:
        return [{"source_id": f"source-{index}", "emotion": "Angry"} for index in range(count)]

    @staticmethod
    def results(sources: list[dict[str, str]]) -> list[dict[str, str]]:
        return [
            {"condition": condition, **source,
             **{f"{channel}_prediction": "angry" for channel in ("clean", "n4", "n8", "n16")}}
            for condition in ("selector", "identity")
            for source in sources
        ]

    def resume(self, rows: list[dict], sources: list[dict]) -> tuple[list[dict], int]:
        with patch.object(Path, "exists", return_value=True), patch("seer.evaluate.load_csv", return_value=rows):
            return resumed_rows(Path("unused.csv"), sources)

    def test_complete_batch_prefix_resumes_without_repeating_sources(self) -> None:
        sources = self.sources(128)
        rows = self.results(sources[:64])
        self.assertEqual(self.resume(rows, sources), (rows, 64))

    def test_complete_short_final_batch_is_accepted(self) -> None:
        sources = self.sources(48)
        rows = self.results(sources)
        self.assertEqual(self.resume(rows, sources), (rows, 48))

    def test_incomplete_batch_or_unpaired_results_are_rejected(self) -> None:
        sources = self.sources(128)
        for rows in (self.results(sources[:63]), self.results(sources[:64])[:-1]):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                self.resume(rows, sources)

    def test_reordered_or_changed_source_results_are_rejected(self) -> None:
        sources = self.sources(64)
        rows = self.results(sources)
        rows[-1], rows[-2] = rows[-2], rows[-1]
        with self.assertRaises(ValueError):
            self.resume(rows, sources)
        rows = self.results(sources)
        rows[0]["emotion"] = "Happy"
        with self.assertRaises(ValueError):
            self.resume(rows, sources)
        rows = self.results(sources)
        rows[0]["n8_prediction"] = ""
        with self.assertRaises(ValueError):
            self.resume(rows, sources)


if __name__ == "__main__":
    unittest.main()
