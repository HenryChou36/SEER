import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np

from data import reference_bank
from seer.ser import EmotionResult, NATIVE_LABELS


def prediction(label):
    scores = np.zeros(len(NATIVE_LABELS), dtype=np.float32)
    scores[NATIVE_LABELS.index(label)] = 1.0
    return EmotionResult(scores=scores, embedding=None)


class ReferenceBankTest(unittest.TestCase):
    def test_bank_build_finishes_with_native_boolean_summary(self):
        config = {
            "emotion_labels": ["Angry", "Happy"],
            "ser_response_labels": ["Neutral", "Angry", "Happy", "Sad"],
            "qualification_views": ["clean", "n8"],
            "source_counts": {"train": 2}, "qualified_references": 1,
        }
        csv = (
            "dataset,speaker_id,utterance_id,split,emotion,emotion2vec_path,processed_audio_path\n"
            "test,1,1,train,Angry,a.npz,a.wav\n"
            "test,1,2,train,Happy,b.npz,b.wav\n"
        )
        evaluator = Mock()
        evaluator.predict.return_value = [prediction("angry"), prediction("neutral")]
        argv = [
            "reference_bank", "--config", "config.json", "--data-root", "input",
            "--output", "output", "--emotion2vec-model", "model", "--device", "cpu",
        ]
        waveform = np.zeros(32, dtype=np.float32)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch.object(Path, "read_text", return_value=json.dumps(config)))
            stack.enter_context(patch.object(Path, "open", mock_open(read_data=csv)))
            stack.enter_context(patch.object(Path, "mkdir"))
            stack.enter_context(patch.object(reference_bank, "load_ser", return_value=evaluator))
            stack.enter_context(patch.object(reference_bank, "_clean_result", side_effect=[prediction("angry"), prediction("happy")]))
            stack.enter_context(patch.object(reference_bank, "_waveform", return_value=waveform))
            stack.enter_context(patch.object(reference_bank, "simulate_many", return_value={8: waveform}))
            writes = stack.enter_context(patch.object(reference_bank, "_write_csv"))
            completion = stack.enter_context(patch.object(reference_bank, "atomic_text"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            reference_bank.main()
        self.assertEqual(writes.call_count, 2)
        self.assertEqual(completion.call_args.args[0].name, "complete.json")
        summary = json.loads(completion.call_args.args[1])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["retained_clean_n8"], 1)
        self.assertEqual(summary["qualified_references"], 1)


if __name__ == "__main__":
    unittest.main()
