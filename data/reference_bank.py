from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from seer.ci import simulate_many
from .common import atomic_text
from seer.ser import EmotionResult, load_ser
from .selector_data import selector_rows


def _source_id(row: dict[str, str]) -> str:
    return f"{row['dataset']}-{row['speaker_id']}-{row['utterance_id']}"


def _clean_result(path: Path) -> EmotionResult:
    with np.load(path, allow_pickle=False) as saved:
        scores = np.asarray(saved["scores"], dtype=np.float32)
    return EmotionResult(scores=scores, embedding=None)


def _waveform(path: Path) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16_000 or values.shape[1] != 1:
        raise ValueError(f"reference qualification requires mono 16-kHz audio: {path}")
    values = values[:, 0]
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"invalid waveform: {path}")
    return values


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    fields = list(rows[0])
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the SER-qualified SEER reference bank"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emotion2vec-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    response_labels = tuple(config["ser_response_labels"])
    qualification_views = tuple(config["qualification_views"])
    allowed_views = {"clean", "n4", "n8", "n16"}
    if not {"clean", "n8"}.issubset(qualification_views) or not set(
        qualification_views
    ).issubset(allowed_views):
        raise ValueError(
            "qualification_views must include clean and n8 and use only clean/n4/n8/n16"
        )
    channel_by_view = {"n4": 4, "n8": 8, "n16": 16}
    ci_views = tuple(view for view in qualification_views if view != "clean")
    ci_channels = tuple(channel_by_view[view] for view in ci_views)
    root = args.data_root.resolve()
    metadata = root / "train.csv"
    with metadata.open(encoding="utf-8", newline="") as handle:
        rows = selector_rows(list(csv.DictReader(handle)), config, split="train")
    expected = int(config["source_counts"]["train"])
    if len(rows) != expected or len({_source_id(row) for row in rows}) != expected:
        raise ValueError("selector training inventory differs from its configuration")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluator = load_ser(
        model=args.emotion2vec_model, device=args.device, batch_size=args.batch_size
    )
    records = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        views: list[np.ndarray] = []
        clean = []
        for row in batch:
            clean.append(_clean_result(root / row["emotion2vec_path"]))
            conditions = simulate_many(
                _waveform(root / row["processed_audio_path"]), ci_channels
            )
            views.extend(conditions[channel] for channel in ci_channels)
        evaluated = evaluator.predict(views, include_embedding=False)
        for position, row in enumerate(batch):
            offset = len(ci_views) * position
            results = {
                "clean": clean[position],
                **dict(
                    zip(ci_views, evaluated[offset : offset + len(ci_views)])
                ),
            }
            projected = {
                view: result.project(response_labels)
                for view, result in results.items()
            }
            predictions = {view: value[0] for view, value in projected.items()}
            scores = {view: value[1] for view, value in projected.items()}
            target = row["emotion"].casefold()
            record = {
                "source_id": _source_id(row),
                **{
                    f"{view}_prediction": predictions[view]
                    for view in qualification_views
                },
                **{
                    f"{view}_scores": json.dumps(scores[view], sort_keys=True)
                    for view in qualification_views
                },
                "own_clean_correct": predictions["clean"] == target,
                "own_n8_correct": predictions["clean"] == target
                and predictions["n8"] == target,
                "qualified_reference": all(
                    predictions[view] == target for view in qualification_views
                ),
            }
            records.append(record)
        print(f"qualified {min(start + len(batch), len(rows)):,}/{len(rows):,}", flush=True)

    joined = [{**row, **record} for row, record in zip(rows, records)]
    qualified = [row for row in joined if row["qualified_reference"]]
    if len(qualified) != int(config["qualified_references"]):
        raise ValueError(
            f"expected {config['qualified_references']} qualified references, "
            f"found {len(qualified)}"
        )
    _write_csv(joined, output / "train_sources.csv")
    _write_csv(qualified, output / "references.csv")
    summary = {
        "complete": True,
        "training_sources": len(joined),
        "qualified_references": len(qualified),
        "qualification_views": list(qualification_views),
        "ser_response_labels": list(response_labels),
        "retained_clean_n8": sum(row["own_n8_correct"] for row in joined),
    }
    atomic_text(output / "complete.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
