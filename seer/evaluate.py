from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from seer.ci import simulate_many
from seer.ser import NATIVE_LABELS, load_ser
from data.selector_data import (
    enhanced_inventory, load_csv, read_waveform, reference_inventory, selector_rows,
    source_id, test_rows,
)
from seer.evc import load_evc
from seer.metrics import f1_by_label, macro_f1
from seer.retriever import Selector, SelectorConfig
from seer.reward import RewardRequest


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def resumed_rows(path: Path, tests: list[dict[str, str]]) -> tuple[list[dict], int]:
    """Accept only complete batches with one result per source and condition."""
    if not path.exists():
        return [], 0
    rows = load_csv(path)
    expected_ids = [source_id(row) for row in tests]
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in ("selector", "identity")
    }
    completed = len(by_condition["selector"])
    if (
        len(rows) != 2 * completed
        or len(by_condition["identity"]) != completed
        or completed > len(tests)
        or (completed % 64 and completed != len(tests))
    ):
        raise ValueError("evaluation progress is not a complete batch prefix")
    for condition_rows in by_condition.values():
        if [row["source_id"] for row in condition_rows] != expected_ids[:completed]:
            raise ValueError("evaluation progress source order differs")
        for result, source in zip(condition_rows, tests[:completed]):
            if result["emotion"].casefold() != source["emotion"].casefold():
                raise ValueError("evaluation progress emotion differs")
            for channel in ("clean", "n4", "n8", "n16"):
                if not result.get(f"{channel}_prediction"):
                    raise ValueError("evaluation progress has a missing prediction")
    return rows, completed


def prepare_resume(args: argparse.Namespace, config: dict) -> None:
    output = args.output.resolve()
    paths = (args.checkpoint, args.maestro_checkpoint, args.emotion2vec_model / "model.pt")
    state = {
        "arguments": {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "config": config,
        "test_csv": (args.data_root / "test.csv").read_text(encoding="utf-8"),
        "references_csv": (args.bank_root / "references.csv").read_text(encoding="utf-8"),
        "maestro_config": args.maestro_config.read_text(encoding="utf-8"),
        "weights": [
            {"path": str(path.resolve()), "size": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        ],
        "evaluation_code": Path(__file__).read_text(encoding="utf-8"),
    }
    if args.enhanced_root is not None:
        enhanced_root = args.enhanced_root.resolve()
        selections = load_csv(enhanced_root / "selections.csv")
        state["enhanced_selections"] = (enhanced_root / "selections.csv").read_text(encoding="utf-8")
        state["enhanced_audio"] = [
            {"path": str((enhanced_root / row["output_audio_path"]).resolve()),
             "size": (enhanced_root / row["output_audio_path"]).stat().st_size,
             "mtime_ns": (enhanced_root / row["output_audio_path"]).stat().st_mtime_ns}
            for row in selections
        ]
    state_path = output / "evaluation_state.json"
    progress_path = output / "predictions.progress.csv"
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")) != state:
            raise ValueError("evaluation inputs differ from saved progress")
    else:
        if progress_path.exists():
            raise ValueError("evaluation progress has no input record")
        output.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(state_path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SEER emotion recognition at Clean/N4/N8/N16"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evc-backend", choices=("maestro",), default="maestro")
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--maestro-checkpoint", type=Path, required=True)
    parser.add_argument("--maestro-config", type=Path, required=True)
    parser.add_argument("--emotion2vec-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--enhanced-root", type=Path,
        help="Score saved enhancement audio after verifying its retriever selections.",
    )
    parser.add_argument("--selector-device")
    parser.add_argument(
        "--source-intensity",
        choices=("normal", "strong"),
        help="Optionally restrict RAVDESS verification to one intensity.",
    )
    parser.add_argument("--reward-device", default="cuda")
    parser.add_argument("--evaluator-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true", help="Continue validated complete evaluation batches.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    labels = tuple(config["emotion_labels"])
    response_labels = tuple(config["ser_response_labels"])
    root = args.data_root.resolve()
    bank_root = args.bank_root.resolve()
    tests = test_rows(load_csv(root / "test.csv"), config, args.source_intensity)
    enhanced = (
        enhanced_inventory(args.enhanced_root, tests)
        if args.enhanced_root is not None else None
    )
    reference_rows = load_csv(bank_root / "references.csv")
    if (
        len(reference_rows) != int(config["qualified_references"])
    ):
        raise ValueError("test or reference inventory differs")
    if len(selector_rows(reference_rows, config, split="train")) != len(reference_rows):
        raise ValueError("reference bank contains rows outside the selector protocol")
    if not tests:
        raise ValueError("evaluation split is empty")
    if args.resume:
        prepare_resume(args, config)
        rows, resume_start = resumed_rows(args.output.resolve() / "predictions.progress.csv", tests)
    else:
        rows, resume_start = [], 0
    references, reference_emotions, reference_paths = reference_inventory(
        reference_rows, root, labels
    )

    selector_device = args.selector_device or str(config["selector_device"])
    if selector_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("selector configuration requires CUDA")
    device = torch.device(selector_device)
    model = Selector(len(labels), SelectorConfig.from_experiment(config)).to(device)
    saved = torch.load(args.checkpoint, map_location="cpu")
    if saved["config"] != config:
        raise ValueError("evaluation configuration differs from the checkpoint")
    model.load_state_dict(saved["model"])
    model.eval()
    references = references.to(device)
    reference_emotions = reference_emotions.to(device)

    evaluator = load_ser(
        model=args.emotion2vec_model,
        device=args.reward_device,
        batch_size=args.evaluator_batch_size,
    )
    evc = None
    if enhanced is None:
        evc = load_evc(
            args.evc_backend,
            upstream_root=args.maestro_root,
            checkpoint=args.maestro_checkpoint,
            config=args.maestro_config,
            device=args.reward_device,
        )

    for start in range(resume_start, len(tests), 64):
        batch = tests[start : start + 64]
        selector_requests, identity_requests, utilities, routed = [], [], [], []
        for row in batch:
            frames = np.load(root / row["emotion_path"], allow_pickle=False)
            if frames.ndim == 3:
                frames = frames[0]
            emotion, action, utility = model.select_reference(
                torch.from_numpy(frames.astype(np.float32, copy=False)),
                references,
                reference_emotions,
            )
            if enhanced is not None:
                selection = enhanced[source_id(row)]
                selected_action = selection["reference_index"]
                expected_action = "" if action is None else str(action)
                if (
                    selected_action != expected_action
                    or selection["predicted_emotion"].casefold() != labels[emotion].casefold()
                    or (action is not None and Path(selection["reference_path"]).resolve() != reference_paths[action].resolve())
                ):
                    raise ValueError(f"saved enhancement selection differs for {source_id(row)}")
            selector_requests.append(
                RewardRequest(
                    source_id(row),
                    root / row["processed_audio_path"],
                    row["emotion"],
                    action,
                    None if action is None else reference_paths[action],
                )
            )
            identity_requests.append(
                RewardRequest(
                    source_id(row),
                    root / row["processed_audio_path"],
                    row["emotion"],
                    None,
                    None,
                )
            )
            utilities.append(utility)
            routed.append(emotion)
        for condition, requests in (
            ("selector", selector_requests),
            ("identity", identity_requests),
        ):
            views = []
            for request in requests:
                if enhanced is not None:
                    waveform = read_waveform(
                        args.enhanced_root / enhanced[request.source_id]["output_audio_path"]
                        if condition == "selector" else request.source_path
                    )
                else:
                    waveform = evc.render(request.source_path, request.reference_path)
                simulated = simulate_many(waveform, (4, 8, 16))
                views.extend((waveform, simulated[4], simulated[8], simulated[16]))
            results = evaluator.predict(views, include_embedding=False)
            for position, (row, request) in enumerate(zip(batch, requests)):
                clean_result, n4_result, n8_result, n16_result = results[
                    4 * position : 4 * position + 4
                ]
                clean_prediction, clean_scores = clean_result.project(response_labels)
                n4_prediction, n4_scores = n4_result.project(response_labels)
                n8_prediction, n8_scores = n8_result.project(response_labels)
                n16_prediction, n16_scores = n16_result.project(response_labels)
                rows.append(
                    {
                        "condition": condition,
                        "source_id": source_id(row),
                        "emotion": row["emotion"],
                        "selected_route": labels[routed[position]],
                        "reference_index": ""
                        if request.reference_index is None
                        else request.reference_index,
                        "reference_path": ""
                        if request.reference_path is None
                        else str(request.reference_path),
                        "predicted_utility": 0.0
                        if condition == "identity"
                        else utilities[position],
                        "clean_prediction": clean_prediction,
                        "n4_prediction": n4_prediction,
                        "n8_prediction": n8_prediction,
                        "n16_prediction": n16_prediction,
                        "clean_native_prediction": NATIVE_LABELS[
                            int(clean_result.scores.argmax())
                        ],
                        "n4_native_prediction": NATIVE_LABELS[int(n4_result.scores.argmax())],
                        "n8_native_prediction": NATIVE_LABELS[int(n8_result.scores.argmax())],
                        "n16_native_prediction": NATIVE_LABELS[int(n16_result.scores.argmax())],
                        "clean_native_scores": json.dumps(clean_result.scores.tolist()),
                        "n4_native_scores": json.dumps(n4_result.scores.tolist()),
                        "n8_native_scores": json.dumps(n8_result.scores.tolist()),
                        "n16_native_scores": json.dumps(n16_result.scores.tolist()),
                        "clean_scores": json.dumps(clean_scores, sort_keys=True),
                        "n4_scores": json.dumps(n4_scores, sort_keys=True),
                        "n8_scores": json.dumps(n8_scores, sort_keys=True),
                        "n16_scores": json.dumps(n16_scores, sort_keys=True),
                    }
                )
        if args.resume:
            write_csv(rows, args.output.resolve() / "predictions.progress.csv")
        print(f"evaluated {min(start + len(batch), len(tests)):,}/{len(tests):,}", flush=True)

    output = args.output.resolve()
    write_csv(rows, output / "predictions.csv")
    metrics = {}
    for condition in ("identity", "selector"):
        selected = [row for row in rows if row["condition"] == condition]
        targets = [row["emotion"].casefold() for row in selected]
        metrics[condition] = {}
        for channel in ("clean", "n4", "n8", "n16"):
            predictions = [row[f"{channel}_prediction"] for row in selected]
            metrics[condition][channel] = {
                "macro_f1": macro_f1(targets, predictions, labels),
                "per_emotion_f1": f1_by_label(targets, predictions, labels),
            }
    identity_rows = {
        str(row["source_id"]): row for row in rows if row["condition"] == "identity"
    }
    selector_by_id = {
        str(row["source_id"]): row for row in rows if row["condition"] == "selector"
    }
    metrics["comparison"] = {}
    for channel in ("clean", "n4", "n8", "n16"):
        repaired = harmed = preserved = still_wrong = 0
        for identifier, identity_row in identity_rows.items():
            selector_row = selector_by_id[identifier]
            target = str(identity_row["emotion"]).casefold()
            before = identity_row[f"{channel}_prediction"] == target
            after = selector_row[f"{channel}_prediction"] == target
            if not before and after:
                repaired += 1
            elif before and not after:
                harmed += 1
            elif before and after:
                preserved += 1
            else:
                still_wrong += 1
        metrics["comparison"][channel] = {
            "repaired": repaired,
            "harmed": harmed,
            "preserved_correct": preserved,
            "still_incorrect": still_wrong,
        }
    selector = [row for row in rows if row["condition"] == "selector"]
    real = [row for row in selector if row["reference_index"] != ""]
    counts: dict[str, int] = {}
    for row in real:
        key = str(row["reference_index"])
        counts[key] = counts.get(key, 0) + 1
    metrics["selection"] = {
        "sources": len(selector),
        "identity_count": len(selector) - len(real),
        "unique_references": len(counts),
        "maximum_reference_count": max(counts.values(), default=0),
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "metrics.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output / "metrics.json")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
