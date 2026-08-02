from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ..data import city, monifeixing, vrq
from ..data.features import (
    assemble_domain_batch,
    domain_batches,
    leave_one_subject_out_logits,
)
from ..evaluation.io import read_csv, read_json, write_json
from ..evaluation.metrics import subject_sort_key
from ..model.a1 import DirectionalMambaKAN
from ..model.femba import FEMBAEncoder
from ..model.main import BioFoundationV1
from ..model.severity import PairSeverityHead


@dataclass(frozen=True)
class SeverityExample:
    subject_id: str
    reference_session: str
    task_session: str
    label: int
    weight: float = 1.0


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def _sample_session_indices(record, session: str, count: int) -> list[int]:
    candidates = [index for index, value in enumerate(record.sessions) if str(value) == session]
    if len(candidates) < count:
        raise ValueError(f"{session} has only {len(candidates)} windows")
    positions = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(np.int64)
    if len(np.unique(positions)) != count:
        raise ValueError(f"{session} cannot provide {count} unique windows")
    return [candidates[int(position)] for position in positions]


def _pair_logits(
    a1: DirectionalMambaKAN,
    head: PairSeverityHead,
    bank,
    examples: list[SeverityExample],
    device: torch.device,
    *,
    windows: int,
) -> torch.Tensor:
    features = []
    for example in examples:
        record = bank.records[example.subject_id]
        reference = _sample_session_indices(record, example.reference_session, windows)
        task = _sample_session_indices(record, example.task_session, windows)
        indices = reference + task
        tokens = torch.as_tensor(record.tokens[indices], dtype=torch.float32, device=device)
        sequence = a1.encode_sequence(tokens)
        normalized = a1.normalize_subject(sequence, sequence)
        embeddings = a1.pool_embedding(normalized)
        features.append(head.features(embeddings[:windows], embeddings[windows:]))
    return head(torch.stack(features))


def _monifeixing_protocol(config: dict, fold_id: str):
    root = (
        Path(config["paths"]["asset_root"])
        / "vr_ssq_regression"
        / "artifacts_fair_joint_lambda0p3"
        / "monifeixing"
        / "lambda0p3"
        / "seed42"
        / "full"
    )
    report = read_json(root / "report.json")
    split = report["identity_audit"]["folds"][fold_id]
    source = list(split["source_outer_train_subjects"])
    labels = {
        str(row["subject_id"]): int(row["y_true"])
        for row in read_csv(root / "severity_predictions.csv")
    }
    bank = monifeixing.build_raw_bank(
        Path(config["paths"]["monifeixing_data_root"]),
        Path(config["paths"]["monifeixing_initial_femba"]),
        torch.device("cpu"),
    )
    return bank, source, [SeverityExample(subject, "rest1", "rest2", labels[subject]) for subject in source]


def run_monifeixing_smoke(config: dict, fold: int, device_name: str, output_root: Path) -> dict:
    fold_id = f"fold_{int(fold)}"
    bank, source_subjects, examples = _monifeixing_protocol(config, fold_id)
    device = torch.device(device_name)
    model = BioFoundationV1(bank.encoder_state, freeze_encoder=True).to(device)
    chosen = source_subjects[:4]
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
    state_logits, _, _, _ = leave_one_subject_out_logits(
        embeddings, label_tensor, model.a1.temperature()
    )
    state_loss = F.binary_cross_entropy_with_logits(state_logits, label_tensor)
    severity_examples = [next(example for example in examples if example.subject_id == subject) for subject in chosen]
    reference_embeddings = [embedding[current_labels < 0.5] for embedding, current_labels in zip(embeddings, label_tensor)]
    task_embeddings = [embedding[current_labels > 0.5] for embedding, current_labels in zip(embeddings, label_tensor)]
    severity_logits = model.severity_logits(reference_embeddings, task_embeddings)
    severity_labels = severity_logits.new_tensor([float(example.label) for example in severity_examples])
    severity_loss = F.binary_cross_entropy_with_logits(severity_logits, severity_labels)
    loss = state_loss + 0.3 * severity_loss
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


def _vrq_training_data(config: dict, fold_id: str, device: torch.device):
    root = (
        Path(config["paths"]["asset_root"])
        / "vr_ssq_regression"
        / "artifacts_fair_joint_lambda0p3"
        / "vrq"
        / "seed_42"
        / "main"
        / "full"
    )
    manifest = read_json(root / "audit_manifest.json")
    payload = manifest["run_fingerprint_payload"]
    args = SimpleNamespace(
        pretrain_ckpt=str(config["paths"]["pretrain_checkpoint"]),
        data_root=str(config["paths"]["vrq_data_root"]),
        mat_key=payload["mat_key"],
        encoder_backend="native",
        ea_mode="subject_unlabeled",
        encode_batch_size=64,
        record_storage_dtype=np.float16,
    )
    protocols = [vrq.SubjectProtocol(**row) for row in manifest["subject_protocols"]]
    bank = vrq.build_feature_bank(args, device, manifest["audit"], protocols)
    split = manifest["folds"][fold_id]
    source = sorted(split["train_subjects"] + split["val_subjects"], key=subject_sort_key)
    task = {row.subject_id: row.final_task for row in protocols}
    examples = [
        SeverityExample(
            subject,
            "rest01",
            task[subject],
            int(manifest["audit"]["subjects"][subject]["ssq_label"]),
        )
        for subject in source
    ]
    return bank, source, examples


def _city_training_data(config: dict, fold_id: str, device: torch.device):
    root = (
        Path(config["paths"]["asset_root"])
        / "vr_ssq_regression"
        / "artifacts_city_a3_lambda_sweep_strict"
        / "audit"
        / "audit_manifest.json"
    )
    manifest = read_json(root)
    audit = json.loads(json.dumps(manifest["audit"]))
    data_root = Path(config["paths"]["city_data_root"])
    for metadata in audit["subjects"].values():
        if metadata.get("included"):
            metadata["mat_path"] = str(data_root / Path(metadata["mat_path"]).name)
    args = SimpleNamespace(
        pretrain_ckpt=str(config["paths"]["pretrain_checkpoint"]),
        encoder_backend="native",
        mat_key="data256",
        ea_mode="subject_unlabeled",
        encode_batch_size=64,
        record_storage_dtype=np.float16,
    )
    bank = city.build_feature_bank(args, device, audit)
    split = manifest["fold_manifest"]["folds"][fold_id]
    source = sorted(split["train_subjects"] + split["val_subjects"], key=subject_sort_key)
    aliases = {}
    for subject, metadata in audit["subjects"].items():
        for segment in metadata.get("segments", []):
            if segment.get("path_score") is not None:
                aliases[(subject, int(segment["route_order"]))] = city.session_alias(
                    segment, metadata["anchor_session"]
                )
    examples = [
        SeverityExample(
            str(row["subject_id"]),
            "rest01",
            aliases[(str(row["subject_id"]), int(row["route_order"]))],
            int(row["path_label"]),
        )
        for row in audit["path_labels"]
        if str(row["subject_id"]) in source
    ]
    return bank, source, examples


def _full_fold_train(
    dataset: str,
    bank,
    source_subjects: list[str],
    examples: list[SeverityExample],
    device: torch.device,
    output_root: Path,
    *,
    epochs: int = 50,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Training refuses a non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1001)
    a1 = DirectionalMambaKAN(0.25).to(device)
    head = PairSeverityHead(a1.embedding_dim, 0.25, 1018).to(device)
    optimizer = torch.optim.AdamW(
        list(a1.parameters()) + list(head.parameters()), lr=1e-4, weight_decay=1e-3
    )
    source_indices = [
        sample.sample_index for sample in bank.samples if sample.subject_id in set(source_subjects)
    ]
    history = []
    rng = random.Random(1001)
    for epoch in range(1, int(epochs) + 1):
        totals = {"loss": 0.0, "state_loss": 0.0, "severity_loss": 0.0, "steps": 0}
        batches = domain_batches(bank, source_indices, 1001 + epoch * 100003, 5, 5)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        for subjects, local_indices in batches:
            tokens, _, labels = assemble_domain_batch(bank, subjects, local_indices)
            tokens = tokens.to(device)
            labels = labels.to(device)
            current = [shuffled[totals["steps"] % len(shuffled)]]
            optimizer.zero_grad(set_to_none=True)
            embeddings = a1.forward_domain_batch(tokens)
            state_logits, _, _, _ = leave_one_subject_out_logits(
                embeddings, labels, a1.temperature()
            )
            state_loss = F.binary_cross_entropy_with_logits(state_logits, labels)
            direct_logits = _pair_logits(a1, head, bank, current, device, windows=5)
            direct_labels = direct_logits.new_tensor([float(example.label) for example in current])
            severity_loss = F.binary_cross_entropy_with_logits(direct_logits, direct_labels)
            loss = state_loss + 0.3 * severity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(a1.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["state_loss"] += float(state_loss.detach().cpu())
            totals["severity_loss"] += float(severity_loss.detach().cpu())
            totals["steps"] += 1
        for key in ("loss", "state_loss", "severity_loss"):
            totals[key] /= max(int(totals["steps"]), 1)
        totals["epoch"] = epoch
        history.append(totals)
        write_json(output_root / "history.json", history)
    torch.save(
        {
            "dataset": dataset,
            "training_seed": 1001,
            "severity_weight": 0.3,
            "encoder_frozen": True,
            "model_spec": a1.model_spec(),
            "model_state_dict": {key: value.detach().cpu() for key, value in a1.state_dict().items()},
            "severity_head_state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()
            },
            "source_subjects": source_subjects,
        },
        output_root / "checkpoint.pt",
    )
    result = {"status": "complete", "dataset": dataset, "epochs": epochs, "history": history}
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
    if dataset == "monifeixing":
        bank, subjects, examples = _monifeixing_protocol(config, fold_id)
        encoder = FEMBAEncoder().to(device)
        encoder.load_state_dict(bank.encoder_state, strict=True)
        monifeixing.refresh_tokens(bank, encoder, subjects, device, 64)
        del encoder
    elif dataset == "vrq":
        bank, subjects, examples = _vrq_training_data(config, fold_id, device)
    elif dataset == "city":
        bank, subjects, examples = _city_training_data(config, fold_id, device)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return _full_fold_train(dataset, bank, subjects, examples, device, output_root)
