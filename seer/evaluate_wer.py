"""Score corpus WER on the same saved SEER audio used for emotion evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import unicodedata

from data.selector_data import enhanced_inventory, load_csv, read_waveform, source_id, test_rows
from seer.ci import simulate_many


SETTINGS = ("clean", "n4", "n8", "n16")


def english_tokens(value: str) -> list[str]:
    text = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^a-z0-9']+", " ", text).split()


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for i, reference_token in enumerate(reference, start=1):
        current = [(i, 0, i, 0)]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            if reference_token == hypothesis_token:
                current.append(previous[j - 1])
                continue
            substitution, deletion, insertion = previous[j - 1], previous[j], current[j - 1]
            current.append(min(
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ))
        previous = current
    return previous[-1][1:]


class Whisper:
    def __init__(self, model: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model), local_files_only=True, torch_dtype=torch.float16
        ).to(device).eval()

    def transcribe(self, waveforms: list) -> list[str]:
        inputs = self.processor(waveforms, sampling_rate=16_000, return_tensors="pt", padding=True)
        features = inputs.input_features.to(self.device, dtype=self.torch.float16)
        with self.torch.inference_mode():
            ids = self.model.generate(
                features, language="en", task="transcribe", do_sample=False,
                num_beams=1, max_new_tokens=128,
            )
        return [text.strip() for text in self.processor.batch_decode(ids, skip_special_tokens=True)]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def summarize(rows: list[dict]) -> dict:
    metrics = {}
    for setting in SETTINGS:
        selected = [row for row in rows if row["setting"] == setting]
        errors = sum(int(row[key]) for row in selected for key in ("substitutions", "deletions", "insertions"))
        words = sum(int(row["reference_words"]) for row in selected)
        if not selected or words <= 0:
            raise ValueError(f"incomplete WER inventory for {setting}")
        metrics[setting] = {
            "sources": len(selected), "errors": errors, "reference_words": words,
            "wer": errors / words, "wer_percent": 100.0 * errors / words,
        }
    return {"selector": metrics}


def resume_rows(path: Path, tasks: list[tuple[dict, str]]) -> list[dict]:
    if not path.exists():
        return []
    rows = load_csv(path)
    if len(rows) > len(tasks):
        raise ValueError("WER progress has more rows than the evaluation inventory")
    for result, (source, setting) in zip(rows, tasks):
        if (
            result["source_id"] != source_id(source) or result["setting"] != setting
            or result["reference_text"] != source["transcript"]
        ):
            raise ValueError("WER progress does not match the requested source/view prefix")
        expected = edit_counts(english_tokens(source["transcript"]), english_tokens(result["hypothesis_text"]))
        recorded = tuple(int(result[key]) for key in ("substitutions", "deletions", "insertions"))
        if recorded != expected or int(result["reference_words"]) != len(english_tokens(source["transcript"])):
            raise ValueError("WER progress word counts differ from its saved transcripts")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--enhanced-root", type=Path, required=True)
    parser.add_argument("--whisper-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-intensity", choices=("normal", "strong"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32, help="Audio views per ASR batch.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    sources = test_rows(load_csv(args.data_root / "test.csv"), config, args.source_intensity)
    selections = enhanced_inventory(args.enhanced_root, sources)
    if any(not english_tokens(row.get("transcript", "")) for row in sources):
        raise ValueError("every test source requires a nonempty ground-truth transcript")
    model = args.whisper_model.resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"missing local Whisper large-v3-turbo model: {model}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(row, setting) for row in sources for setting in SETTINGS]

    progress_path = output / "utterance_wer.progress.csv"
    if args.resume:
        state = {
            "arguments": {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "config": config,
            "test_csv": (args.data_root / "test.csv").read_text(encoding="utf-8"),
            "selections_csv": (args.enhanced_root / "selections.csv").read_text(encoding="utf-8"),
            "files": [
                {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                for path in (
                    *sorted(path for path in model.iterdir() if path.is_file()),
                    *(args.enhanced_root / selections[source_id(row)]["output_audio_path"] for row in sources),
                )
            ],
            "evaluation_code": Path(__file__).read_text(encoding="utf-8"),
        }
        state_path = output / "evaluation_state.json"
        if state_path.exists():
            if json.loads(state_path.read_text(encoding="utf-8")) != state:
                raise ValueError("WER evaluation inputs differ from saved progress")
        else:
            if progress_path.exists():
                raise ValueError("WER progress has no input record")
            atomic_json(state_path, state)
        rows = resume_rows(progress_path, tasks)
    else:
        rows = []

    backend = Whisper(model, args.device) if len(rows) < len(tasks) else None
    for start in range(len(rows), len(tasks), args.batch_size):
        batch = tasks[start : start + args.batch_size]
        views = {}
        for source, _ in batch:
            identifier = source_id(source)
            if identifier not in views:
                waveform = read_waveform(args.enhanced_root / selections[identifier]["output_audio_path"])
                ci = simulate_many(waveform, (4, 8, 16))
                views[identifier] = {"clean": waveform, "n4": ci[4], "n8": ci[8], "n16": ci[16]}
        hypotheses = backend.transcribe([views[source_id(source)][setting] for source, setting in batch])
        if len(hypotheses) != len(batch):
            raise ValueError("Whisper returned an incomplete batch")
        for (source, setting), hypothesis in zip(batch, hypotheses):
            reference = english_tokens(source["transcript"])
            substitutions, deletions, insertions = edit_counts(reference, english_tokens(hypothesis))
            rows.append({
                "source_id": source_id(source), "emotion": source["emotion"], "setting": setting,
                "output_audio_path": selections[source_id(source)]["output_audio_path"],
                "reference_text": source["transcript"], "hypothesis_text": hypothesis,
                "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
                "reference_words": len(reference),
            })
        if args.resume:
            write_csv(progress_path, rows)
        print(f"transcribed {len(rows):,}/{len(tasks):,} audio views", flush=True)

    metrics = summarize(rows)
    write_csv(output / "utterance_wer.csv", rows)
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "complete.json", {
        "complete": True, "sources": len(sources), "views": len(rows), "settings": list(SETTINGS),
        "whisper_model": str(model), "enhanced_root": str(args.enhanced_root.resolve()),
    })
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
