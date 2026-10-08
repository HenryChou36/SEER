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


EMOTIONS = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprise",
}
STATEMENTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}
FOLDS = (
    (4, 12, 13, 23),
    (9, 16, 21, 22),
    (6, 10, 15, 17),
    (7, 14, 18, 19),
    (1, 2, 11, 24),
    (3, 5, 8, 20),
)
MAESTRO_EMOTIONS = {"Neutral", "Angry", "Happy", "Sad", "Surprise"}


def split(actor: int, fold: int) -> str:
    if actor in FOLDS[fold]:
        return "test"
    if actor in FOLDS[(fold + 1) % len(FOLDS)]:
        return "dev"
    return "train"


def scan(root: Path, fold: int) -> list[dict[str, object]]:
    root = root.resolve()
    rows = []
    for path in sorted(root.rglob("*.wav")):
        fields = path.stem.split("-")
        if len(fields) != 7:
            continue
        modality, vocal, emotion, intensity, statement, repetition, actor = fields
        if modality != "03" or vocal != "01":
            continue
        if emotion not in EMOTIONS or intensity not in {"01", "02"}:
            raise ValueError(f"invalid RAVDESS filename: {path.name}")
        actor_number = int(actor)
        outer_fold = next(i for i, actors in enumerate(FOLDS) if actor_number in actors)
        rows.append(
            {
                "dataset": "ravdess",
                "split": split(actor_number, fold),
                "audio_path": path.relative_to(root).as_posix(),
                "processed_name": f"A{actor_number:02d}_{path.name}",
                "utterance_id": path.stem,
                "content_id": f"{statement}-{repetition}",
                "speaker_id": f"A{actor_number:02d}",
                "gender": "M" if actor_number % 2 else "F",
                "language": "English",
                "emotion": EMOTIONS[emotion],
                "intensity": "normal" if intensity == "01" else "strong",
                "outer_fold": outer_fold,
                "statement_id": statement,
                "repetition": repetition,
                "transcript": STATEMENTS[statement],
            }
        )
    if len(rows) != 1440:
        raise ValueError(f"expected 1,440 RAVDESS speech files, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RAVDESS for SEER")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(6), default=0)
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
            "dataset": "ravdess",
            "fold": args.fold,
            "actor_folds": [list(actors) for actors in FOLDS],
            "test_actors": list(FOLDS[args.fold]),
            "dev_actors": list(FOLDS[(args.fold + 1) % len(FOLDS)]),
        },
    )
    rows = process_rows(
        scan(args.root, args.fold),
        args.root,
        output,
        maestro,
        emotion2vec,
        args.metadata_only,
        args.progress_every,
    )
    write_metadata(rows, output / "metadata.csv")
    write_split_metadata(rows, output)
    write_maestro_lists(rows, output, MAESTRO_EMOTIONS)


if __name__ == "__main__":
    main()
