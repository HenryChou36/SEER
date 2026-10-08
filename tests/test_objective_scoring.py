from __future__ import annotations

from argparse import Namespace
from contextlib import ExitStack
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np
import torch

from data.selector_data import enhanced_inventory, test_rows
from seer import enhance, evaluate
from seer.evaluate_wer import SETTINGS, edit_counts, english_tokens, resume_rows, summarize
from seer.ser import EmotionResult


class ObjectivePopulationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "dataset": "ravdess", "emotion_labels": ["Angry", "Happy", "Sad"],
            "source_intensities": ["normal", "strong"], "source_counts": {"test": 96},
        }
        self.sources = [
            {"source_id": str(index), "split": "test", "emotion": self.config["emotion_labels"][index % 3],
             "intensity": "normal" if index % 2 == 0 else "strong"}
            for index in range(96)
        ]

    def test_full_target_test_population_and_normal_subset(self) -> None:
        unrelated = [{"source_id": "neutral", "emotion": "Neutral", "split": "test", "intensity": "normal"}]
        self.assertEqual(len(test_rows(self.sources + unrelated, self.config)), 96)
        normal = test_rows(self.sources + unrelated, self.config, "normal")
        self.assertEqual(len(normal), 48)
        self.assertEqual({row["intensity"] for row in normal}, {"normal"})
        self.assertEqual(test_rows(normal, self.config, "normal"), normal)

    def test_missing_duplicate_or_incomplete_intensity_sources_are_rejected(self) -> None:
        duplicate = self.sources[:-1] + [self.sources[0]]
        changed = [dict(row) for row in self.sources]
        changed[0]["intensity"] = "strong"
        for sources in (self.sources[1:], duplicate, changed):
            with self.subTest(sources=sources[-1]), self.assertRaises(ValueError):
                test_rows(sources, self.config, "normal")

    def test_esd_900_target_sources_reject_an_intensity_filter(self) -> None:
        config = {**self.config, "dataset": "esd_en", "source_intensities": [], "source_counts": {"test": 900}}
        sources = [{"source_id": str(index), "split": "test", "emotion": "Angry", "intensity": ""} for index in range(900)]
        self.assertEqual(len(test_rows(sources, config)), 900)
        with self.assertRaises(ValueError):
            test_rows(sources, config, "normal")

    def test_saved_selections_must_match_sources_and_reference_fallback(self) -> None:
        selection = {
            "source_id": "0", "output_audio_path": "audio/0.wav", "enhanced": "False",
            "reference_index": "", "reference_path": "", "predicted_emotion": "Angry",
        }
        with patch("data.selector_data.load_csv", return_value=[selection]), patch.object(Path, "is_file", return_value=True):
            self.assertEqual(set(enhanced_inventory(Path("export"), self.sources[:1])), {"0"})
            with self.assertRaises(ValueError):
                enhanced_inventory(Path("export"), self.sources[:2])
        for updates in ({"enhanced": "True"}, {"output_audio_path": "../elsewhere.wav"}, {"reference_index": "0"}):
            with patch("data.selector_data.load_csv", return_value=[{**selection, **updates}]), patch.object(Path, "is_file", return_value=True):
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    enhanced_inventory(Path("export"), self.sources[:1])

    def test_enhancer_exports_configured_emotions_and_requested_intensity(self) -> None:
        config = {**self.config, "qualified_references": 1, "selector_device": "cpu"}
        rows = [
            {**self.sources[index], "emotion_path": "feature.npy", "processed_audio_path": "source.wav"}
            for index in range(3)
        ] + [{"source_id": "neutral", "emotion": "Neutral", "intensity": "normal"}]
        args = Namespace(
            config=Path("config.json"), data_root=Path("data"), reference_data_root=None,
            manifest=Path("test.csv"), bank_root=Path("bank"), checkpoint=Path("best.pt"),
            selector_device="cpu", evc_backend="maestro", maestro_root=Path("maestro"),
            maestro_checkpoint=Path("generator.pt"), maestro_config=Path("maestro.json"),
            output=Path("export"), evc_device="cpu", source_intensity=None,
        )
        model = Mock()
        model.select_reference.return_value = (0, None, 0.0)
        for intensity, expected in ((None, ["0", "1", "2"]), ("normal", ["0", "2"])):
            args.source_intensity = intensity
            with (
                patch("argparse.ArgumentParser.parse_args", return_value=args),
                patch.object(Path, "read_text", return_value=json.dumps(config)),
                patch.object(Path, "mkdir"), patch("builtins.print"),
                patch("seer.enhance.load_csv", side_effect=[rows, [{"emotion": "Angry"}]]),
                patch("seer.enhance.reference_inventory", return_value=(torch.zeros(1, 2064), torch.zeros(1, dtype=torch.long), [Path("reference.wav")])),
                patch("seer.enhance.SelectorConfig.from_experiment"),
                patch("seer.enhance.Selector", return_value=Mock(to=Mock(return_value=model))),
                patch("seer.enhance.torch.load", return_value={"config": config, "model": {}}),
                patch("seer.enhance.load_evc"),
                patch("seer.enhance.np.load", return_value=np.zeros((2, 1024), dtype=np.float32)),
                patch("seer.enhance.shutil.copyfile"), patch("seer.enhance.write_csv") as write,
            ):
                enhance.main()
                self.assertEqual([row["source_id"] for row in write.call_args.args[0]], expected)


class WordErrorTest(unittest.TestCase):
    def test_original_normalization_retains_ascii_apostrophes(self) -> None:
        self.assertEqual(english_tokens("ＣＡＮ'T, Stop!  42."), ["can't", "stop", "42"])
        self.assertEqual(english_tokens("Kids are talking by the door."), ["kids", "are", "talking", "by", "the", "door"])

    def test_substitution_deletion_insertion_and_empty_hypothesis(self) -> None:
        self.assertEqual(edit_counts(["a", "b", "c"], ["a", "d", "c", "extra"]), (1, 0, 1))
        self.assertEqual(edit_counts(["a", "b", "c"], ["a", "c"]), (0, 1, 0))
        self.assertEqual(edit_counts(["a", "b"], []), (0, 2, 0))

    def test_corpus_wer_uses_total_errors_over_total_words(self) -> None:
        rows = [
            {"setting": setting, "substitutions": errors, "deletions": 0, "insertions": 0, "reference_words": words}
            for setting in SETTINGS for errors, words in ((1, 1), (0, 9))
        ]
        for setting in SETTINGS:
            metric = summarize(rows)["selector"][setting]
            self.assertEqual(metric["sources"], 2)
            self.assertEqual(metric["wer_percent"], 10.0)

    def test_resume_rejects_reordered_sources_or_tampered_counts(self) -> None:
        source = {"source_id": "source", "transcript": "a b"}
        row = {"source_id": "source", "setting": "clean", "reference_text": "a b", "hypothesis_text": "a", "substitutions": "0", "deletions": "1", "insertions": "0", "reference_words": "2"}
        with patch.object(Path, "exists", return_value=True), patch("seer.evaluate_wer.load_csv", return_value=[row]):
            self.assertEqual(resume_rows(Path("progress.csv"), [(source, "clean")]), [row])
        for updates in ({"source_id": "another"}, {"setting": "n8"}, {"deletions": "0"}):
            with patch.object(Path, "exists", return_value=True), patch("seer.evaluate_wer.load_csv", return_value=[{**row, **updates}]):
                with self.subTest(updates=updates), self.assertRaises(ValueError):
                    resume_rows(Path("progress.csv"), [(source, "clean")])


class SavedWaveformEvaluationTest(unittest.TestCase):
    def test_ser_reads_saved_audio_and_rejects_a_changed_reference_without_loading_evc(self) -> None:
        config = {
            "dataset": "esd_en", "emotion_labels": ["Angry"],
            "ser_response_labels": ["Neutral", "Angry", "Happy", "Sad"],
            "source_counts": {"test": 1}, "qualified_references": 1, "selector_device": "cpu",
        }
        source = {"source_id": "source", "emotion": "Angry", "split": "test", "emotion_path": "frames.npy", "processed_audio_path": "source.wav"}
        reference = {"emotion": "Angry", "split": "train"}
        args = Namespace(
            config=Path("config.json"), data_root=Path("data"), bank_root=Path("bank"),
            checkpoint=Path("best.pt"), maestro_root=Path("maestro"), maestro_checkpoint=Path("generator.pt"),
            maestro_config=Path("maestro.json"), emotion2vec_model=Path("ser"), output=Path("evaluation"),
            enhanced_root=Path("export"), source_intensity=None, selector_device="cpu",
            reward_device="cpu", evaluator_batch_size=64, evc_backend="maestro", resume=False,
        )
        model = Mock()
        model.select_reference.return_value = (0, 0, 1.0)
        native = np.zeros(9, dtype=np.float32)
        native[0] = 1.0
        evaluator = Mock()
        evaluator.predict.return_value = [EmotionResult(native, None)] * 4
        selection = {"source_id": "source", "reference_index": "0", "reference_path": str(Path("data/reference.wav").resolve()), "predicted_emotion": "Angry", "output_audio_path": "audio/source.wav"}
        for index in ("0", "1"):
            with ExitStack() as stack:
                for context in (
                    patch("argparse.ArgumentParser.parse_args", return_value=args),
                    patch.object(Path, "read_text", return_value=json.dumps(config)),
                    patch.object(Path, "open", mock_open()), patch.object(Path, "mkdir"), patch.object(Path, "replace"),
                    patch("seer.evaluate.os.fsync"), patch("builtins.print"), patch("seer.evaluate.write_csv"),
                    patch("seer.evaluate.load_csv", side_effect=[[source], [reference]]),
                    patch("seer.evaluate.enhanced_inventory", return_value={"source": {**selection, "reference_index": index}}),
                    patch("seer.evaluate.reference_inventory", return_value=(torch.zeros(1, 2064), torch.zeros(1, dtype=torch.long), [Path("data/reference.wav").resolve()])),
                    patch("seer.evaluate.SelectorConfig.from_experiment"),
                    patch("seer.evaluate.Selector", return_value=Mock(to=Mock(return_value=model))),
                    patch("seer.evaluate.torch.load", return_value={"config": config, "model": {}}),
                    patch("seer.evaluate.load_ser", return_value=evaluator),
                    patch("seer.evaluate.np.load", return_value=np.zeros((2, 1024), dtype=np.float32)),
                    patch("seer.evaluate.simulate_many", return_value={channel: np.zeros(32, dtype=np.float32) for channel in (4, 8, 16)}),
                ):
                    stack.enter_context(context)
                load_evc = stack.enter_context(patch("seer.evaluate.load_evc"))
                read = stack.enter_context(patch("seer.evaluate.read_waveform", return_value=np.zeros(32, dtype=np.float32)))
                if index == "1":
                    with self.assertRaisesRegex(ValueError, "selection differs"):
                        evaluate.main()
                    read.assert_not_called()
                else:
                    evaluate.main()
                    self.assertEqual(read.call_args_list[0].args[0], Path("export/audio/source.wav"))
                    self.assertEqual(read.call_args_list[1].args[0], Path("data/source.wav").resolve())
                load_evc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
