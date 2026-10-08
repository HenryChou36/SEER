import json
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import DEFAULT, Mock, mock_open, patch

import numpy as np
import torch

from seer.exploration import Decision, Explorer
from seer.reward import ActionOutcome, RewardConfig, RewardGroup, RewardModel
from seer.train import sampled_targets, train


def outcome(correct: bool, confidence: float) -> ActionOutcome:
    return ActionOutcome(
        "angry", "angry" if correct else "neutral", (), (),
        "angry", "angry" if correct else "neutral", {}, {},
        correct, confidence,
    )


class TrainingIntegrationTest(unittest.TestCase):
    def test_trainer_uses_injected_reward_explorer_and_evc(self) -> None:
        config = {
            "emotion_labels": ["Angry"],
            "source_counts": {"train": 2, "dev": 1},
            "qualified_references": 2,
            "seed": 7,
            "hidden_dim": 8,
            "transformer_layers": 1,
            "transformer_heads": 2,
            "transformer_ffn_dim": 16,
            "bandit_hidden_dim": 8,
            "bandit_feature_dim": 4,
            "replay_sample_size": 0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "source_balance_field": "retained",
            "source_batch_size": 2,
            "reference_actions_per_source": 1,
            "gradient_clip": 1.0,
            "maximum_updates": 1,
            "minimum_updates": 1,
            "patience_updates": 2,
            "covariance_rebuild_interval": 1,
            "validation_interval": 2,
        }
        args = Namespace(
            config=Mock(), data_root=Path("data"), bank_root=Path("bank"),
            output=Path("output"), selector_device="cpu",
        )
        args.config.read_text.return_value = json.dumps(config)
        sources = [
            {
                "source_id": f"source-{index}", "emotion": "Angry",
                "split": "train", "retained": retained,
                "processed_audio_path": f"source-{index}.wav",
            }
            for index, retained in enumerate(("True", "False"))
        ]
        validation = [{**sources[0], "split": "dev"}]
        reference_paths = [Path("reference-0.wav"), Path("reference-1.wav")]
        evc = Mock()
        evc.render.return_value = np.zeros(32, dtype=np.float32)
        reward = RewardModel(
            Mock(), RewardConfig(("Angry",), ("Angry", "Neutral"))
        )
        reward.score = Mock(
            return_value=[outcome(False, 0.2), outcome(True, 0.5)] * 2
        )
        reward.utility = Mock(wraps=reward.utility)
        explorer = Mock(spec=Explorer)
        explorer.choose.return_value = Decision(
            0, (1,), torch.zeros(2), torch.zeros(2)
        )
        explorer.observe.side_effect = lambda pairs: tuple(pairs)
        explorer.state_dict.return_value = {"custom_explorer": True}

        with (
            patch.multiple(
                "seer.train",
                load_csv=Mock(side_effect=[sources, validation, sources]),
                reference_inventory=Mock(return_value=(
                    torch.zeros(2, 2064), torch.zeros(2, dtype=torch.long),
                    reference_paths,
                )),
                SourceFeatures=Mock(return_value=[
                    torch.zeros(2, 1024), torch.ones(2, 1024)
                ]),
                validate=DEFAULT,
                save_checkpoint=DEFAULT,
                UCBExplorer=DEFAULT,
                load_evc=DEFAULT,
                load_ser=DEFAULT,
                RewardModel=DEFAULT,
            ) as mocks,
            patch.object(Path, "mkdir"),
            patch.object(Path, "is_file", return_value=False),
            patch.object(Path, "open", mock_open()),
            patch.object(Path, "write_text"),
        ):
            mocks["validate"].return_value = 0.5
            train(args, reward_model=reward, explorer=explorer, evc=evc)

        for name in ("UCBExplorer", "load_evc", "load_ser", "RewardModel"):
            mocks[name].assert_not_called()
        self.assertEqual(explorer.choose.call_count, 2)
        for call in explorer.choose.call_args_list:
            self.assertEqual(call.args[4], 1)
            self.assertEqual(call.kwargs, {"forced_emotion": 0, "explore": True})
        self.assertEqual(evc.render.call_count, 4)
        self.assertEqual(
            [call.args[1] for call in evc.render.call_args_list],
            [None, reference_paths[1], None, reference_paths[1]],
        )
        reward.score.assert_called_once()
        self.assertEqual(reward.utility.call_count, 2)
        explorer.refresh.assert_called_once()
        refreshed = explorer.refresh.call_args.args[0]
        self.assertEqual(refreshed.shape, (2, 4))
        self.assertTrue(torch.isfinite(refreshed).all())
        state = mocks["save_checkpoint"].call_args_list[0].args[1]
        self.assertEqual(state["update"], 1)
        self.assertEqual(state["exploration"], {"custom_explorer": True})
        for record in state["history"]["records"]:
            self.assertEqual(record["reference_index"], 1)
            self.assertAlmostEqual(record["target"], 1.03)
        self.assertIs(mocks["validate"].call_args.args[-2], evc)
        self.assertIs(mocks["validate"].call_args.args[-1], reward)

    def test_sampled_pair_targets_and_cache_reuse(self) -> None:
        evc = Mock()
        evc.render.side_effect = [
            np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32)
        ]
        reward = RewardModel(
            Mock(), RewardConfig(("Angry",), ("Angry", "Neutral"))
        )
        reward.score = Mock(return_value=[outcome(False, 0.2), outcome(True, 0.5)])
        group = RewardGroup(
            "source", Path("source.wav"), "Angry", ((2, Path("reference.wav")),)
        )
        first = sampled_targets(evc, reward, [group])
        self.assertEqual(first[0][0].reference_index, 2)
        self.assertEqual(first[0][0].primary, 1)
        self.assertAlmostEqual(first[0][0].target, 1.03)
        second = sampled_targets(evc, reward, [group])
        self.assertEqual(first, second)
        self.assertEqual(evc.render.call_count, 2)
        self.assertEqual(reward.score.call_args_list[0].args[1], ["Angry", "Angry"])
        self.assertEqual(reward.score.call_count, 1)


if __name__ == "__main__":
    unittest.main()
