"""Apply a trained SEER retriever and write one output waveform per source."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data.selector_data import load_csv, reference_inventory, selector_rows, source_id
from seer.evc import load_evc
from seer.retriever import Selector, SelectorConfig


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return name or "source"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--reference-data-root",
        type=Path,
        help="Preprocessing root that owns the reference-bank feature paths; defaults to data-root.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evc-backend", choices=("maestro",), default="maestro")
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--maestro-checkpoint", type=Path, required=True)
    parser.add_argument("--maestro-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector-device")
    parser.add_argument("--evc-device", default="cuda")
    parser.add_argument("--source-intensity", choices=("normal", "strong"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    labels = tuple(config["emotion_labels"])
    root = args.data_root.resolve()
    reference_root = (
        args.reference_data_root.resolve()
        if args.reference_data_root is not None
        else root
    )
    rows = selector_rows(load_csv(args.manifest.resolve()), config)
    if args.source_intensity:
        if config.get("dataset") != "ravdess":
            parser.error("--source-intensity applies only to RAVDESS")
        rows = [row for row in rows if row.get("intensity", "").casefold() == args.source_intensity]
    if not rows:
        raise ValueError("source manifest is empty")

    reference_rows = load_csv(args.bank_root.resolve() / "references.csv")
    if len(reference_rows) != int(config["qualified_references"]):
        raise ValueError("reference inventory differs from the configuration")
    references, reference_emotions, reference_paths = reference_inventory(
        reference_rows, reference_root, labels
    )

    selector_device = args.selector_device or str(config["selector_device"])
    if selector_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("selector configuration requires CUDA")
    device = torch.device(selector_device)
    model = Selector(len(labels), SelectorConfig.from_experiment(config)).to(device)
    saved = torch.load(args.checkpoint, map_location="cpu")
    if saved["config"] != config:
        raise ValueError("enhancement configuration differs from the checkpoint")
    model.load_state_dict(saved["model"])
    model.eval()
    references = references.to(device)
    reference_emotions = reference_emotions.to(device)
    evc = load_evc(
        args.evc_backend,
        upstream_root=args.maestro_root,
        checkpoint=args.maestro_checkpoint,
        config=args.maestro_config,
        device=args.evc_device,
    )

    output = args.output.resolve()
    audio_output = output / "audio"
    audio_output.mkdir(parents=True, exist_ok=True)
    selections = []
    used_names: set[str] = set()
    for position, row in enumerate(rows):
        identifier = source_id(row)
        name = safe_name(identifier)
        if name in used_names:
            name = f"{name}_{position:06d}"
        used_names.add(name)
        destination = audio_output / f"{name}.wav"

        frames = np.load(root / row["emotion_path"], allow_pickle=False)
        if frames.ndim == 3 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim != 2 or frames.shape[1] != 1024:
            raise ValueError(f"invalid temporal source feature for {identifier}")
        emotion, reference_index, utility = model.select_reference(
            torch.from_numpy(frames.astype(np.float32, copy=False)),
            references,
            reference_emotions,
        )
        source_audio = root / row["processed_audio_path"]
        if reference_index is None:
            shutil.copyfile(source_audio, destination)
            reference_path = ""
        else:
            reference = reference_paths[reference_index]
            waveform = evc.convert(source_audio, reference)
            if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
                raise ValueError(f"invalid EVC output for {identifier}")
            temporary = destination.with_suffix(".tmp.wav")
            sf.write(temporary, waveform, 16_000, subtype="FLOAT")
            temporary.replace(destination)
            reference_path = str(reference)
        selections.append(
            {
                "source_id": identifier,
                "predicted_emotion": labels[emotion],
                "predicted_utility": utility,
                "enhanced": reference_index is not None,
                "reference_index": "" if reference_index is None else reference_index,
                "reference_path": reference_path,
                "output_audio_path": str(destination.relative_to(output)),
            }
        )

    write_csv(selections, output / "selections.csv")
    print(json.dumps({"sources": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
