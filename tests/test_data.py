import unittest
import json
from pathlib import Path

import numpy as np

from data.esd_preprocess import speakers
from data.ravdess_preprocess import FOLDS, split
from data.selector_data import selector_rows
from seer.train import BalancedSources


class DataTest(unittest.TestCase):
    def test_ravdess_folds_partition_actors(self) -> None:
        actors = [actor for fold in FOLDS for actor in fold]
        self.assertEqual(sorted(actors), list(range(1, 25)))
        self.assertEqual(split(4, 0), "test")
        self.assertEqual(split(9, 0), "dev")
        self.assertEqual(split(1, 0), "train")

    def test_esd_language_inventory(self) -> None:
        self.assertEqual(len(speakers("en")), 10)
        self.assertEqual(len(speakers("zh")), 10)
        self.assertEqual(len(speakers("both")), 20)

    def test_selector_protocol_keeps_both_ravdess_intensities(self) -> None:
        rows = [
            {"emotion": "Angry", "intensity": "normal", "split": "train"},
            {"emotion": "Angry", "intensity": "strong", "split": "train"},
            {"emotion": "Happy", "intensity": "normal", "split": "test"},
        ]
        config = {
            "emotion_labels": ["Angry"],
            "source_intensities": ["normal", "strong"],
        }
        selected = selector_rows(rows, config, split="train")
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["intensity"] for row in selected}, {"normal", "strong"})

    def test_reported_reference_bank_protocols(self) -> None:
        root = Path(__file__).resolve().parents[1] / "configs" / "seer"
        ravdess = json.loads((root / "ravdess.json").read_text())
        esd = json.loads((root / "esd.json").read_text())
        self.assertEqual(ravdess["source_intensities"], ["normal", "strong"])
        self.assertEqual(ravdess["qualification_views"], ["clean", "n4", "n8", "n16"])
        self.assertEqual(ravdess["qualified_references"], 281)
        self.assertEqual(ravdess["source_balance_field"], "qualified_reference")
        self.assertEqual(esd["qualification_views"], ["clean", "n8"])
        self.assertEqual(esd["qualified_references"], 6630)
        self.assertEqual(esd["source_balance_field"], "qualified_reference")
        self.assertEqual(
            esd["ser_response_labels"], ["Neutral", "Angry", "Happy", "Sad"]
        )

    def test_training_sampler_balances_the_configured_bank_criterion(self) -> None:
        rows = [
            {"source_id": str(index), "qualified_reference": status}
            for index, status in enumerate(("True", "True", "False", "False"))
        ]
        sampler = BalancedSources(
            rows, np.random.default_rng(7), "qualified_reference"
        )
        selected = [rows[index]["qualified_reference"] for index in sampler.draw(4)]
        self.assertEqual(selected.count("True"), 2)
        self.assertEqual(selected.count("False"), 2)


if __name__ == "__main__":
    unittest.main()
