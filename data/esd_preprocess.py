from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    process_rows,
    write_maestro_lists,
    write_metadata,
    write_split_definition,
    write_split_metadata,
)


EMOTIONS = ("Neutral", "Angry", "Happy", "Sad", "Surprise")
GENDER = {
    "0001": "F", "0002": "F", "0003": "F", "0004": "M", "0005": "M",
    "0006": "M", "0007": "F", "0008": "M", "0009": "F", "0010": "M",
    "0011": "M", "0012": "M", "0013": "M", "0014": "M", "0015": "F",
    "0016": "F", "0017": "F", "0018": "F", "0019": "M", "0020": "M",
}


def speakers(language: str) -> list[tuple[str, str]]:
    if language == "en":
        return [(f"{speaker:04d}", "English") for speaker in range(11, 21)]
    if language == "zh":
        return [(f"{speaker:04d}", "Mandarin") for speaker in range(1, 11)]
    return speakers("zh") + speakers("en")


def annotations(path: Path) -> dict[str, tuple[str, str]]:
    records = {}
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t", 2)
            if len(fields) != 3:
                raise ValueError(f"invalid ESD annotation at {path}:{number}")
            utterance, transcript, emotion = (value.strip() for value in fields)
            records[utterance] = (transcript, emotion)
    if len(records) != 1750:
        raise ValueError(f"expected 1,750 annotations in {path}, found {len(records)}")
    return records


def scan(root: Path, language: str) -> list[dict[str, object]]:
    root = root.resolve()
    rows = []
    for speaker, language_name in speakers(language):
        speaker_root = root / speaker
        labels = annotations(speaker_root / f"{speaker}.txt")
        observed = set()
        for emotion in EMOTIONS:
            paths = sorted((speaker_root / emotion).glob("*.wav"))
            if len(paths) != 350:
                raise ValueError(f"expected 350 files in {speaker_root / emotion}")
            for position, path in enumerate(paths):
                transcript, annotated_emotion = labels[path.stem]
                if annotated_emotion != emotion:
                    raise ValueError(f"ESD label mismatch for {path}")
                rows.append(
                    {
                        "dataset": "esd_en" if language_name == "English" else "esd_zh",
                        "split": "dev" if position < 20 else "test" if position < 50 else "train",
                        "audio_path": path.relative_to(root).as_posix(),
                        "processed_name": path.name,
                        "utterance_id": path.stem,
                        "content_id": f"{position + 1:04d}",
                        "speaker_id": speaker,
                        "gender": GENDER[speaker],
                        "language": language_name,
                        "emotion": emotion,
                        "intensity": "",
                        "outer_fold": "",
                        "statement_id": "",
                        "repetition": "",
                        "transcript": transcript,
                    }
                )
                observed.add(path.stem)
        if observed != set(labels):
            raise ValueError(f"ESD annotation/audio mismatch for speaker {speaker}")
    expected = 17500 if language in {"en", "zh"} else 35000
    if len(rows) != expected:
        raise ValueError(f"expected {expected:,} ESD files, found {len(rows):,}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ESD for SEER")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "zh", "both"), default="en")
    parser.add_argument("--maestro-root", type=Path)
    parser.add_argument("--emotion-diarization-model", type=Path)
    parser.add_argument("--speaker-model", type=Path)
    parser.add_argument("--emotion2vec-model", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    maestro = emotion2vec = None
    if not args.metadata_only:
        required = (
            args.maestro_root,
            args.emotion_diarization_model,
            args.speaker_model,
            args.emotion2vec_model,
            args.feature_cache,
        )
        if any(value is None for value in required):
            parser.error("feature extraction requires all model/root arguments")
        from seer.ser import load_ser
        from .maestro_features import MaestroFeatures

        maestro = MaestroFeatures(
            args.maestro_root,
            args.emotion_diarization_model,
            args.speaker_model,
            args.device,
            args.feature_cache,
        )
        emotion2vec = load_ser(model=args.emotion2vec_model, device=args.device)

    output = args.output.resolve()
    write_split_definition(
        output,
        {
            "dataset": "esd",
            "language": args.language,
            "speakers": [speaker for speaker, _ in speakers(args.language)],
            "within_speaker_emotion_positions": {
                "dev": [0, 20],
                "test": [20, 50],
                "train": [50, 350],
            },
        },
    )
    rows = process_rows(
        scan(args.root, args.language),
        args.root,
        output,
        maestro,
        emotion2vec,
        args.metadata_only,
        args.progress_every,
    )
    write_metadata(rows, output / "metadata.csv")
    write_split_metadata(rows, output)
    write_maestro_lists(rows, output, set(EMOTIONS))


if __name__ == "__main__":
    main()
