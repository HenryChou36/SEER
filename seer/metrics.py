from __future__ import annotations

import numpy as np


def f1_by_label(
    targets: list[str], predictions: list[str], labels: tuple[str, ...]
) -> dict[str, float]:
    result = {}
    for original in labels:
        label = original.casefold()
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        result[original] = 0.0 if not denominator else 2 * tp / denominator
    return result


def macro_f1(targets: list[str], predictions: list[str], labels: tuple[str, ...]) -> float:
    return float(np.mean(list(f1_by_label(targets, predictions, labels).values())))
