import unittest

import numpy as np

from seer.ci import CIConfig, simulate, simulate_many


class CISimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        time = np.arange(4_000, dtype=np.float64) / 16_000
        self.audio = (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)

    def test_conditions_share_length_and_are_deterministic(self) -> None:
        outputs = simulate_many(self.audio, (4, 8, 16))
        self.assertEqual(set(outputs), {4, 8, 16})
        self.assertTrue(all(value.shape == self.audio.shape for value in outputs.values()))
        np.testing.assert_array_equal(outputs[8], simulate(self.audio, 8))

    def test_invalid_n_of_m_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate(self.audio, CIConfig().available_channels + 1)


if __name__ == "__main__":
    unittest.main()
