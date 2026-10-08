from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from seer.exploration import Explorer, UCBExplorer
from data.selector_data import (
    SourceFeatures,
    csv_bool,
    load_csv,
    reference_inventory,
    selector_rows,
    source_id,
)
from seer.evc import EVC, load_evc
from seer.metrics import macro_f1
from seer.retriever import Selector, SelectorConfig
from seer.reward import (
    ActionOutcome,
    EvaluatedPair,
    EvaluationHistory,
    RewardConfig,
    RewardGroup,
    RewardModel,
    RewardRequest,
    RewardTarget,
)
from seer.ser import load_ser


def evaluate_pairs(
    evc: EVC, reward: RewardModel, requests: list[RewardRequest]
) -> list[ActionOutcome]:
    """Render only uncached pairs, then let the reward model score their audio."""
    outcomes = [reward.cache.get(request) for request in requests]
    missing = [index for index, outcome in enumerate(outcomes) if outcome is None]
    waveforms = []
    for index in missing:
        request = requests[index]
        if (request.reference_index is None) != (request.reference_path is None):
            raise ValueError("reference index and path must both be present or absent")
        waveforms.append(evc.render(request.source_path, request.reference_path))
    evaluated = (
        reward.score(waveforms, [requests[index].intended_emotion for index in missing])
        if missing else []
    )
    if len(evaluated) != len(missing):
        raise ValueError("reward model returned an incomplete batch")
    committed = []
    for index, outcome in zip(missing, evaluated):
        outcomes[index] = outcome
        committed.append((requests[index], outcome))
    reward.cache.add(committed)
    if any(outcome is None for outcome in outcomes):
        raise AssertionError("reward batch is incomplete")
    return [outcome for outcome in outcomes if outcome is not None]


def sampled_targets(
    evc: EVC, reward: RewardModel, groups: list[RewardGroup]
) -> list[list[RewardTarget]]:
    """Compare each sampled conversion with its original speech outcome."""
    requests = []
    for group in groups:
        requests.append(
            RewardRequest(
                group.source_id, group.source_path, group.intended_emotion, None, None
            )
        )
        requests.extend(
            RewardRequest(
                group.source_id, group.source_path, group.intended_emotion, index, path
            )
            for index, path in group.references
        )
    outcomes = evaluate_pairs(evc, reward, requests)
    result = []
    cursor = 0
    for group in groups:
        original = outcomes[cursor]
        cursor += 1
        targets = []
        for index, path in group.references:
            converted = outcomes[cursor]
            cursor += 1
            target, primary, secondary = reward.utility(original, converted)
            targets.append(
                RewardTarget(index, path, target, primary, secondary, converted)
            )
        result.append(targets)
    return result


class BalancedSources:
    def __init__(
        self,
        rows: list[dict[str, str]],
        rng: np.random.Generator,
        status_field: str,
    ) -> None:
        self.rng = rng
        if not status_field or any(status_field not in row for row in rows):
            raise ValueError("balanced source status is absent")
        self.groups = {
            retained: np.asarray(
                sorted(
                    (
                        index
                        for index, row in enumerate(rows)
                        if csv_bool(row[status_field]) == retained
                    ),
                    key=lambda index: source_id(rows[index]),
                ),
                dtype=np.int64,
            )
            for retained in (False, True)
        }
        if any(not len(group) for group in self.groups.values()):
            raise ValueError("balanced sampling requires retained and lost sources")
        self.orders = {key: self.rng.permutation(value) for key, value in self.groups.items()}
        self.batch_index = 0
        self.batch_size: int | None = None

    def draw(self, batch_size: int) -> list[int]:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("balanced source batch size must be positive and even")
        if self.batch_size is not None and self.batch_size != batch_size:
            raise ValueError("balanced source batch size changed")
        self.batch_size = batch_size
        half = batch_size // 2
        cycle_length = (max(map(len, self.groups.values())) + half - 1) // half
        if self.batch_index == cycle_length:
            self.orders = {
                key: self.rng.permutation(value) for key, value in self.groups.items()
            }
            self.batch_index = 0
        result = []
        for order in self.orders.values():
            positions = (self.batch_index * half + np.arange(half)) % len(order)
            result.extend(order[positions].tolist())
        self.batch_index += 1
        return [int(value) for value in self.rng.permutation(result)]

    def state_dict(self) -> dict[str, object]:
        return {
            "orders": {str(key): value.tolist() for key, value in self.orders.items()},
            "batch_index": self.batch_index,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        for key in self.groups:
            label = str(key)
            order = np.asarray(state["orders"][label], dtype=np.int64)
            if sorted(order.tolist()) != sorted(self.groups[key].tolist()):
                raise ValueError("source-sampler checkpoint inventory differs")
            self.orders[key] = order
        self.batch_index = int(state["batch_index"])
        self.batch_size = (
            None if state["batch_size"] is None else int(state["batch_size"])
        )


def optimize(
    model: Selector,
    optimizer: torch.optim.Optimizer,
    records: list[EvaluatedPair],
    current_sources: set[int],
    sources: list[dict[str, str]],
    source_features: SourceFeatures,
    references: torch.Tensor,
    emotion_indices: dict[str, int],
    gradient_clip: float,
) -> tuple[float, float]:
    grouped: dict[int, list[EvaluatedPair]] = defaultdict(list)
    for record in records:
        grouped[record.source_index].append(record)
    predicted, expected, route_losses = [], [], []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for index, group in grouped.items():
        summary = model.encode_source(source_features[index])
        action_indices = torch.tensor(
            [record.reference_index for record in group], device=model.device
        )
        means, _ = model.score_pairs(summary, references[action_indices])
        predicted.append(means)
        expected.append(
            torch.tensor(
                [record.target for record in group],
                device=model.device,
                dtype=torch.float32,
            )
        )
        if index in current_sources:
            target = torch.tensor(
                [emotion_indices[sources[index]["emotion"].casefold()]],
                device=model.device,
            )
            route_losses.append(F.cross_entropy(model.router(summary).unsqueeze(0), target))
    utility_loss = F.mse_loss(torch.cat(predicted), torch.cat(expected))
    route_loss = torch.stack(route_losses).mean()
    loss = utility_loss + route_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    return float(utility_loss.detach().cpu()), float(route_loss.detach().cpu())


@torch.no_grad()
def refresh_explorer(
    model: Selector,
    explorer: Explorer,
    history: EvaluationHistory,
    active_keys: tuple[tuple[int, int], ...],
    source_features: SourceFeatures,
    references: torch.Tensor,
) -> None:
    grouped: dict[int, list[int]] = defaultdict(list)
    for record in history.records(active_keys):
        grouped[record.source_index].append(record.reference_index)
    features = []
    model.eval()
    for source_index, actions in grouped.items():
        summary = model.encode_source(source_features[source_index])
        indices = torch.tensor(actions, device=model.device)
        features.append(model.pair_features(summary, references[indices]))
    if not features:
        raise ValueError("cannot refresh exploration from empty history")
    explorer.refresh(torch.cat(features))


@torch.no_grad()
def validate(
    model: Selector,
    rows: list[dict[str, str]],
    root: Path,
    references: torch.Tensor,
    reference_emotions: torch.Tensor,
    reference_paths: list[Path],
    evc: EVC,
    reward: RewardModel,
) -> float:
    model.eval()
    predictions, targets = [], []
    for offset in range(0, len(rows), 64):
        batch = rows[offset : offset + 64]
        requests = []
        for row in batch:
            frames = np.load(root / row["emotion_path"], allow_pickle=False)
            if frames.ndim == 3:
                frames = frames[0]
            _, action, _ = model.select_reference(
                torch.from_numpy(frames.astype(np.float32, copy=False)),
                references,
                reference_emotions,
            )
            requests.append(
                RewardRequest(
                    source_id(row),
                    root / row["processed_audio_path"],
                    row["emotion"],
                    action,
                    None if action is None else reference_paths[action],
                )
            )
        outcomes = evaluate_pairs(evc, reward, requests)
        predictions.extend(outcome.ci_prediction for outcome in outcomes)
        targets.extend(row["emotion"].casefold() for row in batch)
    return macro_f1(targets, predictions, reward.config.target_labels)


def save_checkpoint(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one SEER retriever with uncertainty-guided acquisition"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--evc-backend", choices=("maestro",), default="maestro")
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--maestro-checkpoint", type=Path, required=True)
    parser.add_argument("--maestro-config", type=Path, required=True)
    parser.add_argument("--emotion2vec-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector-device")
    parser.add_argument("--reward-device", default="cuda")
    parser.add_argument("--evaluator-batch-size", type=int, default=64)
    return parser.parse_args()


def train(
    args: argparse.Namespace,
    *,
    reward_model: RewardModel | None = None,
    explorer: Explorer | None = None,
    evc: EVC | None = None,
) -> None:
    """Train with supplied reward and exploration components or their defaults."""

    config = json.loads(args.config.read_text(encoding="utf-8"))
    labels = tuple(config["emotion_labels"])
    label_indices = {label.casefold(): index for index, label in enumerate(labels)}
    root = args.data_root.resolve()
    bank_root = args.bank_root.resolve()
    sources = load_csv(bank_root / "train_sources.csv")
    validation = selector_rows(load_csv(root / "dev.csv"), config, split="dev")
    reference_rows = load_csv(bank_root / "references.csv")
    if (
        len(sources) != int(config["source_counts"]["train"])
        or len(validation) != int(config["source_counts"]["dev"])
        or len(reference_rows) != int(config["qualified_references"])
    ):
        raise ValueError("training, validation, or reference inventory differs")
    if len(selector_rows(sources, config, split="train")) != len(sources):
        raise ValueError("training bank contains rows outside the selector protocol")
    if len(selector_rows(reference_rows, config, split="train")) != len(reference_rows):
        raise ValueError("reference bank contains rows outside the selector protocol")
    references, reference_emotions, reference_paths = reference_inventory(
        reference_rows, root, labels
    )
    rng = np.random.default_rng(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    selector_device = args.selector_device or str(config["selector_device"])
    if selector_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("selector configuration requires CUDA")
    device = torch.device(selector_device)
    model = Selector(len(labels), SelectorConfig.from_experiment(config)).to(device)
    references = references.to(device)
    reference_emotions = reference_emotions.to(device)
    if explorer is None:
        explorer = UCBExplorer(
            int(config["bandit_feature_dim"]),
            float(config["linucb_alpha"]),
            float(config["linucb_lambda"]),
            device,
            history_capacity=(
                int(config["source_batch_size"])
                * int(config["reference_actions_per_source"])
                * int(config["exploration_history_batches"])
            ),
        )
    history = EvaluationHistory()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    sampler = BalancedSources(sources, rng, str(config["source_balance_field"]))
    features = SourceFeatures(root, sources)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if evc is None:
        evc = load_evc(
            args.evc_backend,
            upstream_root=args.maestro_root,
            checkpoint=args.maestro_checkpoint,
            config=args.maestro_config,
            device=args.reward_device,
        )
    if reward_model is None:
        evaluator = load_ser(
            model=args.emotion2vec_model,
            device=args.reward_device,
            batch_size=args.evaluator_batch_size,
        )
        reward_model = RewardModel(
            evaluator,
            RewardConfig(
                target_labels=labels,
                response_labels=tuple(config["ser_response_labels"]),
                selected_channels=int(config["training_channel"]),
                tie_scale=float(config["tie_scale"]),
            ),
            cache_path=output / "rewards.jsonl",
        )
    reward = reward_model

    latest = output / "latest.pt"
    update = 0
    best_f1 = -1.0
    best_update = 0
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu")
        if saved["config"] != config:
            raise ValueError("training checkpoint configuration differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        explorer.load_state_dict(saved["exploration"], device)
        history.load_state_dict(saved["history"])
        sampler.load_state_dict(saved["sampler"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        update = int(saved["update"])
        best_f1 = float(saved["best_f1"])
        best_update = int(saved["best_update"])

    while update < int(config["maximum_updates"]):
        indices = sampler.draw(int(config["source_batch_size"]))
        groups, decisions = [], []
        model.eval()
        with torch.no_grad():
            for index in indices:
                row = sources[index]
                emotion = label_indices[row["emotion"].casefold()]
                decision = explorer.choose(
                    model,
                    features[index],
                    references,
                    reference_emotions,
                    int(config["reference_actions_per_source"]),
                    forced_emotion=emotion,
                    explore=True,
                )
                decisions.append((index, decision.reference_indices))
                groups.append(
                    RewardGroup(
                        source_id(row),
                        root / row["processed_audio_path"],
                        row["emotion"],
                        tuple(
                            (action, reference_paths[action])
                            for action in decision.reference_indices
                        ),
                    )
                )
        rewards = sampled_targets(evc, reward, groups)
        current = []
        for (index, actions), targets in zip(decisions, rewards):
            if tuple(target.reference_index for target in targets) != actions:
                raise ValueError("reward actions differ from sampled actions")
            current.extend(
                EvaluatedPair(index, target.reference_index, target.target, target.primary)
                for target in targets
            )
        history.add(current)
        keys = {(record.source_index, record.reference_index) for record in current}
        active_keys = explorer.observe(
            (record.source_index, record.reference_index) for record in current
        )
        historical = history.sample(
            int(config["replay_sample_size"]), rng, keys, keys=active_keys
        )
        utility_loss, route_loss = optimize(
            model,
            optimizer,
            current + historical,
            set(indices),
            sources,
            features,
            references,
            label_indices,
            float(config["gradient_clip"]),
        )
        update += 1
        if update == 1 or update % int(config["covariance_rebuild_interval"]) == 0:
            refresh_explorer(model, explorer, history, active_keys, features, references)

        if (
            update % int(config["validation_interval"]) == 0
            or update == int(config["maximum_updates"])
        ):
            score = validate(
                model,
                validation,
                root,
                references,
                reference_emotions,
                reference_paths,
                evc,
                reward,
            )
            improved = score > best_f1
            if improved:
                best_f1, best_update = score, update
            state = {
                "config": config,
                "update": update,
                "best_f1": best_f1,
                "best_update": best_update,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "exploration": explorer.state_dict(),
                "history": history.state_dict(),
                "sampler": sampler.state_dict(),
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
            }
            save_checkpoint(latest, state)
            if improved:
                save_checkpoint(output / "best.pt", state)
            with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "update": update,
                            "n8_macro_f1": score,
                            "utility_loss": utility_loss,
                            "router_loss": route_loss,
                            "history_size": len(history.records()),
                        }
                    )
                    + "\n"
                )
            print(
                f"update={update} n8_macro_f1={score:.4f} "
                f"utility_loss={utility_loss:.4f} router_loss={route_loss:.4f}",
                flush=True,
            )
            if (
                update >= int(config["minimum_updates"])
                and update - best_update >= int(config["patience_updates"])
            ):
                break

    (output / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "last_update": update,
                "best_update": best_update,
                "best_n8_macro_f1": best_f1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
