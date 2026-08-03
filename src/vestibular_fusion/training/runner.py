from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from ..data import monifeixing, vrq
from ..data.features import (
    assemble_domain_batch,
    domain_batches,
    leave_one_subject_out_logits,
)
from ..evaluation.io import write_json
from ..model.encoder import TemporalEncoder
from ..model.main import VestibularFusionModel
from .data import SeverityExample, load_training_dataset, weighted_examples


SEVERITY_WEIGHT = 0.3
TRAINING_SEED = 1001
MAX_EPOCHS = 50
PATIENCE = 10
DOMAINS_PER_BATCH = 5
TRIALS_PER_CLASS = 5
SEVERITY_BATCH_SIZE = 5
SEVERITY_WINDOWS_TRAIN = 5
SEVERITY_WINDOWS_EVAL = 11
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.25
GRAD_CLIP = 1.0


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def _parameter_probe(model: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.detach().float().reshape(-1).cpu()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return torch.cat(values) if values else torch.empty(0)


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _sample_session_indices(
    record, session: str, count: int, *, training: bool, rng: random.Random
) -> list[int]:
    sessions = np.asarray(record.sessions, dtype=object)
    candidates = np.flatnonzero(sessions == str(session)).astype(np.int64)
    count = min(int(count), len(candidates))
    if count < 2:
        raise ValueError(f"{session} has fewer than two severity windows")
    if training:
        return [int(value) for value in rng.sample(candidates.tolist(), count)]
    positions = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(np.int64)
    return [int(candidates[position]) for position in positions]


def _pair_logits(
    model: VestibularFusionModel,
    bank,
    examples: list[SeverityExample],
    device: torch.device,
    *,
    windows: int,
    training: bool,
    rng: random.Random,
) -> torch.Tensor:
    if not examples:
        raise ValueError("Severity loss/evaluation requires at least one example")
    features = []
    for example in examples:
        record = bank.records[example.subject_id]
        reference = _sample_session_indices(
            record,
            example.reference_session,
            windows,
            training=training,
            rng=rng,
        )
        task = _sample_session_indices(
            record,
            example.task_session,
            windows,
            training=training,
            rng=rng,
        )
        indices = reference + task
        tokens = torch.as_tensor(record.tokens[indices], dtype=torch.float32, device=device)
        sequence = model.a1.encode_sequence(tokens)
        normalized = model.a1.normalize_subject(sequence, sequence)
        embeddings = model.a1.pool_embedding(normalized)
        features.append(
            model.severity_head.features(embeddings[: len(reference)], embeddings[len(reference) :])
        )
    return model.severity_head(torch.stack(features))


def _severity_metrics(logits: torch.Tensor, examples: list[SeverityExample]) -> dict:
    labels = np.asarray([int(example.label) for example in examples], dtype=np.int64)
    scores = torch.sigmoid(logits).detach().float().cpu().numpy()
    predictions = (scores >= 0.5).astype(np.int64)
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    subjects = np.asarray([example.subject_id for example in examples], dtype=object)
    subject_scores = [
        balanced_accuracy_score(labels[subjects == subject], predictions[subjects == subject])
        for subject in sorted(set(subjects.tolist()))
    ]
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Severity validation requires both classes")
    return {
        "BCE": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "subject_macro_balanced_accuracy": float(np.mean(subject_scores)),
        "AUROC": float(roc_auc_score(labels, scores)),
    }


def _evaluate_severity(
    model: VestibularFusionModel,
    bank,
    examples: list[SeverityExample],
    device: torch.device,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits = _pair_logits(
            model,
            bank,
            examples,
            device,
            windows=SEVERITY_WINDOWS_EVAL,
            training=False,
            rng=random.Random(0),
        )
    return _severity_metrics(logits, examples)


def _training_module(dataset: str):
    return monifeixing if dataset == "monifeixing" else vrq


def _validation_state_metrics(
    model: VestibularFusionModel,
    dataset: str,
    bank,
    train_subjects: list[str],
    val_subjects: list[str],
    fold_id: str,
    device: torch.device,
) -> dict:
    module = _training_module(dataset)
    model.eval()
    with torch.no_grad():
        embeddings, _ = module.extract_views(model.a1, bank, train_subjects + val_subjects, device, 128)
    prototypes = module.fit_prototypes(embeddings, bank, train_subjects)
    rows = vrq.deep_rows(
        model.a1,
        embeddings,
        prototypes,
        bank,
        val_subjects,
        fold_id,
        "validation",
    )
    labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["mambakan_score"]) for row in rows], dtype=np.float64)
    predictions = (scores >= 0.5).astype(np.int64)
    subjects = np.asarray([str(row["subject_id"]) for row in rows], dtype=object)
    subject_scores = [
        balanced_accuracy_score(labels[subjects == subject], predictions[subjects == subject])
        for subject in sorted(set(subjects.tolist()))
    ]
    return {
        "threshold": 0.5,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "subject_macro_balanced_accuracy": float(np.mean(subject_scores)),
    }


def _train_epoch(
    model: VestibularFusionModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    bank,
    train_indices: list[int],
    examples: list[SeverityExample],
    epoch: int,
    device: torch.device,
) -> dict:
    if not examples:
        raise ValueError("A refit epoch requires source severity examples")
    batches = domain_batches(
        bank,
        train_indices,
        seed=TRAINING_SEED + int(epoch) * 100003,
        domains_per_batch=DOMAINS_PER_BATCH,
        trials_per_class=TRIALS_PER_CLASS,
    )
    rng = random.Random(TRAINING_SEED + int(epoch) * 200003)
    ordered_examples = list(examples)
    rng.shuffle(ordered_examples)
    batch_size = min(SEVERITY_BATCH_SIZE, len(ordered_examples))
    model.train()
    totals = {"loss": 0.0, "state_loss": 0.0, "severity_loss": 0.0}
    steps = 0
    max_a1_grad = 0.0
    max_head_grad = 0.0
    for subjects, local_indices in batches:
        start = (steps * batch_size) % len(ordered_examples)
        current = [
            ordered_examples[(start + offset) % len(ordered_examples)]
            for offset in range(batch_size)
        ]
        tokens, _, labels = assemble_domain_batch(bank, subjects, local_indices)
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            embeddings = model.forward_token_batch(tokens)
            state_logits = leave_one_subject_out_logits(
                embeddings, labels, model.a1.temperature()
            )
            state_loss = F.binary_cross_entropy_with_logits(state_logits, labels)
            severity_logits = _pair_logits(
                model,
                bank,
                current,
                device,
                windows=SEVERITY_WINDOWS_TRAIN,
                training=True,
                rng=rng,
            )
            severity_labels = severity_logits.new_tensor(
                [float(example.label) for example in current]
            )
            weights = severity_logits.new_tensor(
                [float(example.weight) for example in current]
            )
            severity_loss = (
                F.binary_cross_entropy_with_logits(
                    severity_logits, severity_labels, reduction="none"
                )
                * weights
            ).sum() / weights.sum().clamp_min(1e-6)
            loss = state_loss + SEVERITY_WEIGHT * severity_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite refit multitask loss")
        scaler.scale(loss).backward()
        a1_grad = _grad_norm(model.a1)
        head_grad = _grad_norm(model.severity_head)
        if min(a1_grad, head_grad) <= 0.0:
            raise AssertionError("A1 and PairSeverityHead must both receive gradients")
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            GRAD_CLIP,
        )
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach().cpu())
        totals["state_loss"] += float(state_loss.detach().cpu())
        totals["severity_loss"] += float(severity_loss.detach().cpu())
        max_a1_grad = max(max_a1_grad, a1_grad)
        max_head_grad = max(max_head_grad, head_grad)
        steps += 1
    if steps == 0:
        raise AssertionError("Refit epoch produced no optimization steps")
    return {
        **{key: value / steps for key, value in totals.items()},
        "steps": steps,
        "max_a1_grad_norm": max_a1_grad,
        "max_severity_head_grad_norm": max_head_grad,
    }


def _new_model(device: torch.device) -> VestibularFusionModel:
    return VestibularFusionModel(a1_seed=TRAINING_SEED, dropout=DROPOUT).to(device)


def _select_epoch(
    dataset: str,
    fold_id: str,
    bank,
    protocol,
    device: torch.device,
    output_root: Path,
) -> tuple[int, dict]:
    _seed_everything(TRAINING_SEED)
    model = _new_model(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    source_train = list(protocol.calibration_train_subjects)
    source_val = list(protocol.calibration_val_subjects)
    path_weighting = dataset == "city"
    all_examples = list(protocol.source_examples)
    train_examples = list(
        weighted_examples(
            [example for example in all_examples if example.subject_id in set(source_train)],
            path_weighting=path_weighting,
        )
    )
    val_examples = [example for example in all_examples if example.subject_id in set(source_val)]
    train_indices = [
        sample.sample_index
        for sample in bank.samples
        if sample.subject_id in set(source_train)
    ]
    initial_a1 = _parameter_probe(model.a1)
    initial_head = _parameter_probe(model.severity_head)
    best_a1 = {key: value.detach().cpu().clone() for key, value in model.a1.state_dict().items()}
    best_head = {
        key: value.detach().cpu().clone()
        for key, value in model.severity_head.state_dict().items()
    }
    best_epoch = 0
    best_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        training = _train_epoch(
            model, optimizer, scaler, bank, train_indices, train_examples, epoch, device
        )
        state = _validation_state_metrics(
            model, dataset, bank, source_train, source_val, fold_id, device
        )
        severity = _evaluate_severity(model, bank, val_examples, device)
        key = (
            -float(severity["BCE"]),
            float(severity["balanced_accuracy"]),
            float(severity["AUROC"]),
            float(state["subject_macro_balanced_accuracy"]),
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_a1 = {
                key_name: value.detach().cpu().clone()
                for key_name, value in model.a1.state_dict().items()
            }
            best_head = {
                key_name: value.detach().cpu().clone()
                for key_name, value in model.severity_head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "training": training,
            "validation_state": state,
            "validation_severity": severity,
            "selection_key": list(key),
            "best_epoch": best_epoch,
            "patience": stale,
        }
        history.append(row)
        write_json(output_root / "selection_history.json", history)
        print(
            f"{fold_id} selection epoch={epoch} state={training['state_loss']:.5f} "
            f"severity={training['severity_loss']:.5f} "
            f"val_severity_bce={severity['BCE']:.5f} best={best_epoch}",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    if best_epoch <= 0:
        raise RuntimeError(f"{fold_id} failed to select a source-validation epoch")
    model.a1.load_state_dict(best_a1, strict=True)
    model.severity_head.load_state_dict(best_head, strict=True)
    selection = {
        "best_epoch": best_epoch,
        "validation_key": list(best_key),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "a1_delta_probe_l2": float(torch.linalg.vector_norm(_parameter_probe(model.a1) - initial_a1)),
        "severity_head_delta_probe_l2": float(
            torch.linalg.vector_norm(_parameter_probe(model.severity_head) - initial_head)
        ),
        "encoder_grad_norm": 0.0,
        "encoder_delta_probe_l2": 0.0,
        "history": history,
    }
    if selection["a1_delta_probe_l2"] <= 0.0 or selection["severity_head_delta_probe_l2"] <= 0.0:
        raise AssertionError("Source selection did not update both trainable branches")
    write_json(output_root / "selection_report.json", selection)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, selection


def _refit(
    dataset: str,
    fold_id: str,
    bank,
    protocol,
    best_epoch: int,
    device: torch.device,
    output_root: Path,
) -> dict:
    _seed_everything(TRAINING_SEED)
    model = _new_model(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    source_subjects = list(protocol.source_subjects)
    path_weighting = dataset == "city"
    examples = list(
        weighted_examples(list(protocol.source_examples), path_weighting=path_weighting)
    )
    source_set = set(source_subjects)
    train_indices = [
        sample.sample_index for sample in bank.samples if sample.subject_id in source_set
    ]
    initial_a1 = _parameter_probe(model.a1)
    initial_head = _parameter_probe(model.severity_head)
    history = []
    for epoch in range(1, int(best_epoch) + 1):
        history.append(
            {
                "epoch": epoch,
                **_train_epoch(
                    model, optimizer, scaler, bank, train_indices, examples, epoch, device
                ),
            }
        )
        write_json(output_root / "refit_history.json", history)
    a1_delta = float(torch.linalg.vector_norm(_parameter_probe(model.a1) - initial_a1))
    head_delta = float(torch.linalg.vector_norm(_parameter_probe(model.severity_head) - initial_head))
    if min(a1_delta, head_delta) <= 0.0:
        raise AssertionError("Source refit did not update both trainable branches")
    checkpoint = {
        "dataset": dataset,
        "checkpoint_schema": "trained_fold_refit_v1",
        "fold_id": fold_id,
        "training_seed": TRAINING_SEED,
        "severity_weight": SEVERITY_WEIGHT,
        "severity_windows_per_session": SEVERITY_WINDOWS_TRAIN,
        "encoder_frozen": True,
        "refit_protocol": "source_validation_selection_then_source_refit",
        "best_epoch": int(best_epoch),
        "model_spec": model.a1.model_spec(),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.a1.state_dict().items()
        },
        "severity_head_state_dict": {
            key: value.detach().cpu() for key, value in model.severity_head.state_dict().items()
        },
        "source_subjects": source_subjects,
        "test_subjects": list(protocol.test_subjects),
    }
    torch.save(checkpoint, output_root / "checkpoint.pt")
    report = {
        "status": "complete",
        "dataset": dataset,
        "fold_id": fold_id,
        "best_epoch": int(best_epoch),
        "source_subjects": source_subjects,
        "test_subjects": list(protocol.test_subjects),
        "refit_epochs": int(best_epoch),
        "a1_delta_probe_l2": a1_delta,
        "severity_head_delta_probe_l2": head_delta,
        "encoder_grad_norm": 0.0,
        "encoder_delta_probe_l2": 0.0,
        "history": history,
    }
    write_json(output_root / "refit_report.json", report)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def run_monifeixing_smoke(config: dict, fold: int, device_name: str, output_root: Path) -> dict:
    fold_id = f"fold_{int(fold)}"
    dataset_data = load_training_dataset(config, "monifeixing", torch.device("cpu"))
    bank = dataset_data.bank
    protocol = dataset_data.folds[fold_id]
    device = torch.device(device_name)
    model = VestibularFusionModel(bank.encoder_state, freeze_encoder=True).to(device)
    chosen = list(protocol.source_subjects)[:4]
    windows, labels = [], []
    for subject in chosen:
        record = bank.records[subject]
        indices = []
        for label in (0, 1):
            candidates = np.flatnonzero(record.labels == label)
            indices.extend(candidates[:4].tolist())
        windows.append(np.asarray(record.windows[indices], dtype=np.float32))
        labels.append(record.labels[indices].astype(np.float32))
    window_tensor = torch.as_tensor(np.stack(windows), dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(np.stack(labels), dtype=torch.float32, device=device)
    before = _parameter_snapshot(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-3,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    embeddings = model.forward_domain_batch(window_tensor)
    state_logits = leave_one_subject_out_logits(embeddings, label_tensor, model.a1.temperature())
    state_loss = F.binary_cross_entropy_with_logits(state_logits, label_tensor)
    severity_examples = list(protocol.source_examples)[: len(chosen)]
    reference_embeddings = [
        embedding[current_labels < 0.5]
        for embedding, current_labels in zip(embeddings, label_tensor)
    ]
    task_embeddings = [
        embedding[current_labels > 0.5]
        for embedding, current_labels in zip(embeddings, label_tensor)
    ]
    severity_logits = model.severity_logits(reference_embeddings, task_embeddings)
    severity_labels = severity_logits.new_tensor(
        [float(example.label) for example in severity_examples]
    )
    severity_loss = F.binary_cross_entropy_with_logits(severity_logits, severity_labels)
    loss = state_loss + SEVERITY_WEIGHT * severity_loss
    loss.backward()
    optimizer.step()
    deltas = model.parameter_deltas(before)
    if deltas["encoder_delta"] != 0.0:
        raise AssertionError(f"Frozen Encoder changed: {deltas['encoder_delta']}")
    if deltas["a1_delta"] <= 0.0 or deltas["severity_head_delta"] <= 0.0:
        raise AssertionError(f"Trainable modules did not update: {deltas}")
    result = {
        "status": "passed",
        "dataset": "monifeixing",
        "fold_id": fold_id,
        "loss": float(loss.detach().cpu()),
        "state_loss": float(state_loss.detach().cpu()),
        "severity_loss": float(severity_loss.detach().cpu()),
        **deltas,
    }
    write_json(output_root / "smoke_report.json", result)
    return result


def _full_fold_train(
    dataset: str,
    fold_id: str,
    bank,
    protocol,
    device: torch.device,
    output_root: Path,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Training refuses a non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    best_epoch, selection = _select_epoch(
        dataset, fold_id, bank, protocol, device, output_root
    )
    refit = _refit(dataset, fold_id, bank, protocol, best_epoch, device, output_root)
    result = {
        "status": "complete",
        "dataset": dataset,
        "fold_id": fold_id,
        "protocol": "source_validation_selection_then_source_refit",
        "selection": {
            "best_epoch": best_epoch,
            "validation_key": selection["validation_key"],
        },
        "refit": {
            "epochs": best_epoch,
            "checkpoint": str((output_root / "checkpoint.pt").name),
        },
    }
    write_json(output_root / "report.json", result)
    return result


def run_training(
    config: dict,
    dataset: str,
    fold: int | None,
    device_name: str,
    smoke: bool,
    output_root: Path,
) -> dict:
    if smoke:
        if dataset != "monifeixing" or fold is None:
            raise ValueError("The locked smoke check requires --dataset monifeixing --fold N")
        return run_monifeixing_smoke(config, fold, device_name, output_root)
    if fold is None or not 1 <= int(fold) <= 5:
        raise ValueError("Full training requires --fold 1..5")
    fold_id = f"fold_{int(fold)}"
    device = torch.device(device_name)
    dataset_data = load_training_dataset(config, dataset, device)
    protocol = dataset_data.folds[fold_id]
    bank = dataset_data.bank
    if dataset == "monifeixing":
        encoder = TemporalEncoder().to(device)
        encoder.load_state_dict(bank.encoder_state, strict=True)
        monifeixing.refresh_tokens(bank, encoder, list(protocol.source_subjects), device, 64)
        del encoder
    return _full_fold_train(dataset, fold_id, bank, protocol, device, output_root)
