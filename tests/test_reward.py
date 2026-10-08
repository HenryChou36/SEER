import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from seer.reward import (
    ActionOutcome,
    EvaluatedPair,
    EvaluationHistory,
    RewardConfig,
    RewardModel,
)
from seer.ser import Emotion2Vec, EmotionResult, NATIVE_LABELS, clear_ser_cache, load_ser


class FakeEvaluator:
    def __init__(self, results=()) -> None:
        self.results = list(results)
        self.views = []

    def predict(self, views, include_embedding=True):
        self.views = list(views)
        if include_embedding:
            raise AssertionError("reward evaluation does not need embeddings")
        return self.results


def outcome(correct: bool, score: float) -> ActionOutcome:
    return ActionOutcome(
        clean_native_prediction="angry",
        ci_native_prediction="angry" if correct else "happy",
        clean_native_scores=(0.8, 0.2),
        ci_native_scores=(score, 1.0 - score),
        clean_prediction="angry",
        ci_prediction="angry" if correct else "happy",
        clean_scores={"angry": 0.8, "happy": 0.2},
        ci_scores={"angry": score, "happy": 1.0 - score},
        jointly_correct=correct,
        gated_score=score,
    )


def emotion_scores(**values: float) -> EmotionResult:
    scores = np.zeros(len(NATIVE_LABELS), dtype=np.float32)
    for label, value in values.items():
        scores[NATIVE_LABELS.index(label)] = value
    return EmotionResult(scores, None)


class SERTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_ser_cache()
        self.addCleanup(clear_ser_cache)

    def test_loader_reuses_instances_and_distinguishes_model_settings(self) -> None:
        with patch("seer.ser.Emotion2Vec", side_effect=lambda *args, **kwargs: object()) as factory:
            first = load_ser(model=Path("model"), device="cpu", batch_size=8)
            self.assertIs(
                first,
                load_ser("EMOTION2VEC", model=Path("./model"), device="cpu", batch_size=8),
            )
            others = [
                load_ser(model=Path("other"), device="cpu", batch_size=8),
                load_ser(model=Path("model"), device="cuda", batch_size=8),
                load_ser(model=Path("model"), device="cpu", batch_size=16),
            ]
            self.assertTrue(all(instance is not first for instance in others))
            self.assertEqual(factory.call_count, 4)
            factory.assert_any_call(Path("model").resolve(), device="cpu", batch_size=8)
            clear_ser_cache()
            self.assertIsNot(first, load_ser(model=Path("model"), device="cpu", batch_size=8))
            self.assertEqual(factory.call_count, 5)
            with self.assertRaisesRegex(ValueError, "unsupported SER backend"):
                load_ser("unknown", model=Path("model"))
            self.assertEqual(factory.call_count, 5)

    def test_external_evaluator_checks_model_shape_without_reading_weights(self) -> None:
        external = SimpleNamespace(
            kwargs={"tokenizer": SimpleNamespace(token_list=list(NATIVE_LABELS))},
            model=SimpleNamespace(proj=SimpleNamespace(in_features=1024, out_features=9)),
        )
        factory = Mock(return_value=external)
        with (
            patch.dict("sys.modules", {"funasr": SimpleNamespace(AutoModel=factory)}),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "open", side_effect=AssertionError("routine weight read")),
        ):
            evaluator = Emotion2Vec(Path("emotion2vec_plus_large"), device="cpu")
            self.assertIs(evaluator.model, external)
            external.model.proj.in_features = 768
            with self.assertRaisesRegex(ValueError, "1024-D nine-class head"):
                Emotion2Vec(Path("emotion2vec_plus_large"), device="cpu")


class RewardTest(unittest.TestCase):
    def test_original_relative_utility(self) -> None:
        model = RewardModel(
            FakeEvaluator(),
            RewardConfig(
                ("angry", "happy"),
                ("neutral", "angry", "happy", "sad"),
                tie_scale=0.1,
            ),
        )
        target, primary, secondary = model.utility(outcome(False, 0.4), outcome(True, 0.7))
        self.assertEqual(primary, 1)
        self.assertAlmostEqual(secondary, 0.3)
        self.assertAlmostEqual(target, 1.03)
        reverse = model.utility(outcome(True, 0.7), outcome(False, 0.4))
        self.assertAlmostEqual(reverse[0], -1.03)
        self.assertEqual(reverse[1], -1)
        tie = model.utility(outcome(True, 0.7), outcome(True, 0.9))
        self.assertAlmostEqual(tie[0], 0.02)
        self.assertEqual(tie[1], 0)

    def test_score_pairs_clean_and_ci_views_and_gates_confidence(self) -> None:
        evaluator = FakeEvaluator(
            (
                emotion_scores(fearful=0.6, angry=0.24, neutral=0.16),
                emotion_scores(fearful=0.6, angry=0.26, neutral=0.14),
                emotion_scores(angry=0.4, happy=0.6),
                emotion_scores(angry=0.9, happy=0.1),
                emotion_scores(angry=0.6, neutral=0.4),
                emotion_scores(angry=0.1, neutral=0.7, happy=0.2),
            )
        )
        model = RewardModel(
            evaluator,
            RewardConfig(("angry", "happy", "sad"), ("angry", "happy", "sad", "neutral")),
        )
        waveforms = [np.ones(256, dtype=np.float32) * value for value in (0.1, 0.2, 0.3)]
        with patch("seer.reward.simulate", side_effect=lambda audio, count, config: audio * 2):
            outcomes = model.score(waveforms, ("Angry", "angry", "angry"))
        for index, waveform in enumerate(waveforms):
            np.testing.assert_array_equal(evaluator.views[2 * index], waveform)
            np.testing.assert_array_equal(evaluator.views[2 * index + 1], waveform * 2)
        self.assertEqual(outcomes[0].clean_native_prediction, "fearful")
        self.assertEqual(outcomes[0].clean_prediction, "angry")
        self.assertTrue(outcomes[0].jointly_correct)
        self.assertAlmostEqual(outcomes[0].gated_score, 0.65)
        self.assertFalse(outcomes[1].jointly_correct)
        self.assertEqual(outcomes[1].gated_score, 0.0)
        self.assertFalse(outcomes[2].jointly_correct)
        self.assertAlmostEqual(outcomes[2].gated_score, 0.1)
        self.assertAlmostEqual(sum(outcomes[0].clean_scores.values()), 1.0)

    def test_score_rejects_misaligned_labels(self) -> None:
        model = RewardModel(
            FakeEvaluator(),
            RewardConfig(("angry",), ("angry", "neutral")),
        )
        with self.assertRaisesRegex(ValueError, "every reward waveform"):
            model.score([np.ones(256, dtype=np.float32)], [])
        with self.assertRaisesRegex(ValueError, "outside the enhancement targets"):
            model.score([np.ones(256, dtype=np.float32)], ["neutral"])

    def test_neutral_is_a_reward_response_not_an_enhancement_target(self) -> None:
        scores = np.zeros(len(NATIVE_LABELS), dtype=np.float32)
        scores[NATIVE_LABELS.index("neutral")] = 0.6
        scores[NATIVE_LABELS.index("angry")] = 0.4
        prediction, projected = EmotionResult(scores, None).project(
            ("neutral", "angry", "happy", "sad")
        )
        self.assertEqual(prediction, "neutral")
        self.assertAlmostEqual(sum(projected.values()), 1.0)


class EvaluationHistoryTest(unittest.TestCase):
    def test_all_distinct_feedback_survives_duplicates_and_restart(self) -> None:
        records = [
            EvaluatedPair(index, 0, float(index % 3 - 1), index % 3 - 1)
            for index in range(16_385)
        ]
        history = EvaluationHistory()
        history.add(records)
        history.add(records[:10])
        self.assertEqual(history.records(), records)
        restored = EvaluationHistory()
        restored.load_state_dict(history.state_dict())
        self.assertEqual(restored.records(), records)

    def test_cached_pair_outcomes_are_immutable(self) -> None:
        record = EvaluatedPair(0, 1, 0.2, 0)
        history = EvaluationHistory()
        history.add([record])
        for changed in (EvaluatedPair(0, 1, 0.3, 0), EvaluatedPair(0, 1, 0.2, 1)):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                ValueError, "immutable evaluated-pair outcome changed"
            ):
                history.add([changed])
        self.assertEqual(history.records(), [record])

    def test_history_sampling_balances_outcomes_and_excludes_current_pairs(self) -> None:
        history = EvaluationHistory()
        records = [
            EvaluatedPair(index, 0, float(index % 3 - 1), index % 3 - 1)
            for index in range(12)
        ]
        history.add(records)
        selected = history.sample(6, np.random.default_rng(7), {(0, 0)})
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(set(selected)), 6)
        self.assertNotIn(records[0], selected)
        self.assertEqual(
            [sum(record.primary == value for record in selected) for value in (-1, 0, 1)],
            [2, 2, 2],
        )
        self.assertEqual(history.records(), records)

    def test_learning_keys_restrict_sampling_without_discarding_the_archive(self) -> None:
        history = EvaluationHistory()
        records = [
            EvaluatedPair(index, 0, float(index % 3 - 1), index % 3 - 1)
            for index in range(12)
        ]
        history.add(records)
        keys = [(index, 0) for index in range(11, 2, -1)]
        self.assertEqual(history.records(keys), records[11:2:-1])
        selected = history.sample(6, np.random.default_rng(7), {(3, 0)}, keys=keys)
        self.assertEqual(len(set(selected)), 6)
        self.assertTrue(all((record.source_index, record.reference_index) in keys for record in selected))
        self.assertNotIn(records[3], selected)
        self.assertEqual(
            [sum(record.primary == value for record in selected) for value in (-1, 0, 1)],
            [2, 2, 2],
        )
        self.assertEqual(history.records(), records)


if __name__ == "__main__":
    unittest.main()
