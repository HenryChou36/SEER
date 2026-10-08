# Audio examples

These examples compare Source, Strong, JD-NMF, and SEER speech from RAVDESS. One example is provided for each emotion: Angry, Happy, and Sad. Within each emotion, the recordings share the same speaker and sentence.

| Filename prefix | Speech version |
|---|---|
| `source` | Original normal-intensity speech |
| `strong` | Strong-intensity speech |
| `jdnmf` | Speech converted using JD-NMF |
| `seer` | Speech enhanced using SEER |

Each example includes clean audio and three CI-simulated versions. The suffixes `n4`, `n8`, and `n16` indicate selection of 4, 8, and 16 channels from the same 22-channel bank. For example, compare `source_n8.wav` with `seer_n8.wav` to hear the effect of SEER under N8.

The examples were selected from our listening study to illustrate cases where SEER improved emotion recognition across all three CI conditions. They use the original experiment audio, with no additional volume adjustment.

## Attribution

The recordings come from [RAVDESS, Livingstone and Russo (2018)](https://doi.org/10.5281/zenodo.1188976), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The converted and CI-simulated audio are derivatives of these recordings and are distributed under the same license.
