import unittest
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn.functional as F

from seer.exploration import UCBExplorer
from seer.retriever import Selector, SelectorConfig
from seer.train import BalancedSources


class SelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        config = SelectorConfig(
            source_dim=8,
            reference_dim=12,
            hidden_dim=8,
            transformer_layers=1,
            transformer_heads=2,
            transformer_ffn_dim=16,
            pair_hidden_dim=8,
            pair_feature_dim=4,
        )
        self.model = Selector(emotion_count=3, config=config)
        self.frames = torch.randn(5, 8)
        self.references = torch.randn(6, 12)
        self.reference_emotions = torch.tensor([0, 0, 1, 1, 2, 2])
        self.explorer = UCBExplorer(feature_dim=4)

    def test_model_is_independent_of_the_candidate_inventory(self) -> None:
        output = self.model(self.frames, self.references[:3])
        self.assertEqual(output["route_logits"].shape, (3,))
        self.assertEqual(output["utilities"].shape, (3,))
        self.assertEqual(output["pair_features"].shape, (3, 4))

    def test_sampler_returns_distinct_routed_references(self) -> None:
        decision = self.explorer.choose(
            self.model,
            self.frames,
            self.references,
            self.reference_emotions,
            count=2,
            forced_emotion=1,
        )
        self.assertEqual(set(decision.reference_indices), {2, 3})

    def test_identity_is_the_fixed_zero_inference_action(self) -> None:
        with torch.no_grad():
            self.model.utility.weight.zero_()
            self.model.utility.bias.fill_(-1.0)
        _, action, utility = self.model.select_reference(
            self.frames, self.references, self.reference_emotions
        )
        self.assertIsNone(action)
        self.assertEqual(utility, -1.0)

    def test_enhancement_selects_the_best_positive_utility_without_exploration(self) -> None:
        with torch.no_grad():
            self.model.utility.bias.fill_(2.0)
        output = self.model(self.frames, self.references)
        emotion = int(output["route_logits"].argmax())
        eligible = torch.nonzero(self.reference_emotions == emotion).flatten()
        best = int(eligible[torch.topk(output["utilities"][eligible], 1).indices[0]])
        predicted_emotion, action, utility = self.model.select_reference(
            self.frames, self.references, self.reference_emotions
        )
        self.assertEqual(predicted_emotion, emotion)
        self.assertEqual(action, best)
        self.assertGreater(utility, 0.0)
        self.explorer.covariance.fill_(float("nan"))
        self.assertEqual(
            self.model.select_reference(
                self.frames, self.references, self.reference_emotions
            ),
            (predicted_emotion, action, utility),
        )

    def test_covariance_lives_outside_the_model(self) -> None:
        source = self.model.encode_source(self.frames)
        features = self.model.pair_features(source, self.references[:2])
        before = self.explorer.covariance.clone()
        self.explorer.refresh(features)
        self.assertFalse(torch.equal(before, self.explorer.covariance))

    def test_utility_and_auxiliary_emotion_objectives_update_the_retriever(self) -> None:
        output = self.model(self.frames, self.references)
        loss = F.mse_loss(output["utilities"], torch.zeros(6)) + F.cross_entropy(
            output["route_logits"].unsqueeze(0), torch.tensor([2])
        )
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))


class ExplorationHistoryTest(unittest.TestCase):
    def test_observed_pairs_keep_insertion_order_without_duplicate_entries(self) -> None:
        explorer = UCBExplorer(feature_dim=4)
        self.assertEqual(
            explorer.observe(iter([(1, 2), (3, 4), (1, 2)])),
            ((1, 2), (3, 4)),
        )
        self.assertEqual(
            explorer.observe([(3, 4), (5, 6)]),
            ((1, 2), (3, 4), (5, 6)),
        )

    def test_fifo_window_preserves_duplicate_age_and_allows_reentry(self) -> None:
        explorer = UCBExplorer(feature_dim=4, history_capacity=3)
        explorer.observe([(0, 0), (1, 1), (2, 2)])
        self.assertEqual(
            explorer.observe([(0, 0), (3, 3)]),
            ((1, 1), (2, 2), (3, 3)),
        )
        self.assertEqual(
            explorer.observe([(0, 0)]),
            ((2, 2), (3, 3), (0, 0)),
        )

    def test_resume_restores_window_order_and_covariance(self) -> None:
        explorer = UCBExplorer(feature_dim=4, history_capacity=3)
        explorer.observe([(0, 0), (1, 1), (2, 2), (3, 3)])
        explorer.refresh(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
        state = explorer.state_dict()
        resumed = UCBExplorer(feature_dim=4, history_capacity=3)
        resumed.load_state_dict(state, torch.device("cpu"))
        self.assertEqual(resumed.observe([]), ((1, 1), (2, 2), (3, 3)))
        torch.testing.assert_close(resumed.covariance, explorer.covariance)
        continuation = [(1, 1), (0, 0), (4, 4)]
        self.assertEqual(resumed.observe(continuation), explorer.observe(continuation))

    def test_resume_rejects_a_different_window_capacity(self) -> None:
        state = UCBExplorer(feature_dim=4, history_capacity=3).state_dict()
        resumed = UCBExplorer(feature_dim=4, history_capacity=4)
        with self.assertRaisesRegex(ValueError, "checkpoint configuration differs"):
            resumed.load_state_dict(state, torch.device("cpu"))


class SourceSamplingTest(unittest.TestCase):
    @staticmethod
    def sources() -> list[dict[str, str]]:
        return [
            {"source_id": f"source-{index:02}", "retained": str(index >= 3)}
            for index in range(10)
        ]

    def test_smaller_source_stratum_repeats_until_the_shared_cycle_ends(self) -> None:
        rng = Mock(spec=np.random.Generator)
        minority_permutations = 0

        def permutation(values):
            nonlocal minority_permutations
            values = np.asarray(values)
            if len(values) == 3:
                minority_permutations += 1
                return values if minority_permutations == 1 else values[::-1]
            return values.copy()

        rng.permutation.side_effect = permutation
        sampler = BalancedSources(self.sources(), rng, "retained")
        batches = [sampler.draw(4) for _ in range(4)]
        for batch in batches:
            self.assertEqual(len(set(batch)), 4)
            self.assertEqual(sum(index < 3 for index in batch), 2)
        minority = [index for batch in batches for index in batch if index < 3]
        majority = [index for batch in batches for index in batch if index >= 3]
        self.assertEqual(minority, [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(len(set(majority[:7])), 7)
        self.assertEqual(majority[7], majority[0])
        self.assertEqual([index for index in sampler.draw(4) if index < 3], [2, 1])

    def test_source_sampling_resumes_across_shared_cycle_boundaries(self) -> None:
        rng = np.random.default_rng(7)
        sampler = BalancedSources(self.sources(), rng, "retained")
        for _ in range(3):
            sampler.draw(4)
        state, rng_state = sampler.state_dict(), rng.bit_generator.state
        expected = [sampler.draw(4) for _ in range(8)]
        resumed_rng = np.random.default_rng(99)
        resumed = BalancedSources(self.sources(), resumed_rng, "retained")
        resumed.load_state_dict(state)
        resumed_rng.bit_generator.state = rng_state
        self.assertEqual([resumed.draw(4) for _ in range(8)], expected)


if __name__ == "__main__":
    unittest.main()
