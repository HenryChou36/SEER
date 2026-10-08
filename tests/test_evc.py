import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from seer.evc import EVC, clear_evc_cache, load_evc


class FakeBackend:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def convert(self, source, reference):
        self.calls.append((source, reference))
        return np.array([0.1, -0.2], dtype=np.float32)


class EVCTest(unittest.TestCase):
    def setUp(self):
        clear_evc_cache()

    def tearDown(self):
        clear_evc_cache()

    def load(self, checkpoint="checkpoint.pt", device="cpu"):
        return load_evc(
            "maestro",
            upstream_root=Path("upstream"),
            checkpoint=Path(checkpoint),
            config=Path("config.json"),
            device=device,
        )

    def test_one_model_load_serves_repeated_requests_and_conversions(self):
        with patch("seer.evc.MaestroEVC", side_effect=FakeBackend) as loader:
            first = self.load()
            second = self.load("unused/../checkpoint.pt")
            self.assertIs(first, second)
            first(Path("source.wav"), Path("reference-a.wav"))
            second.render(Path("source.wav"), Path("reference-b.wav"))
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(len(first.backend.calls), 2)

    def test_model_specification_and_device_select_separate_instances(self):
        with patch("seer.evc.MaestroEVC", side_effect=FakeBackend) as loader:
            first = self.load()
            self.assertIsNot(first, self.load("another-checkpoint.pt"))
            self.assertIsNot(first, self.load(device="cuda:0"))
            self.assertEqual(loader.call_count, 3)

    def test_render_original_keeps_sample_levels_and_does_not_convert(self):
        model = EVC(FakeBackend())
        samples = np.array([[0.01], [-0.03], [0.02]], dtype=np.float32)
        with patch("seer.evc.sf.read", return_value=(samples, 16_000)):
            actual = model.render(Path("source.wav"))
        np.testing.assert_array_equal(actual, samples[:, 0])
        self.assertEqual(model.backend.calls, [])

    def test_invalid_original_audio_is_rejected(self):
        model = EVC(FakeBackend())
        for samples, sample_rate in [
            (np.ones((2, 1), dtype=np.float32), 8_000),
            (np.empty((0, 1), dtype=np.float32), 16_000),
            (np.array([[np.nan]], dtype=np.float32), 16_000),
        ]:
            with self.subTest(sample_rate=sample_rate, samples=samples):
                with patch("seer.evc.sf.read", return_value=(samples, sample_rate)):
                    with self.assertRaises(ValueError):
                        model.render(Path("source.wav"))

    def test_unknown_backend_is_rejected_without_loading_a_model(self):
        with patch("seer.evc.MaestroEVC") as loader:
            with self.assertRaisesRegex(ValueError, "unsupported EVC backend"):
                load_evc(
                    "unknown",
                    upstream_root=Path("upstream"),
                    checkpoint=Path("checkpoint.pt"),
                    config=Path("config.json"),
                )
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
