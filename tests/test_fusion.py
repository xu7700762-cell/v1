import numpy as np
import torch

from vestibular_fusion.data.features import leave_one_subject_out_logits
from vestibular_fusion.evaluation.fusion import fuse_state_evidence, uniform_anchor_mask
from vestibular_fusion.evaluation.severity import subject_r4_features, winsorized_std
from vestibular_fusion.model.a1 import load_checkpoint_state_dict


def test_state_fusion_is_r1_r2_plus_two_r4_over_four():
    evidence = np.asarray([[1.0, 2.0, 3.0], [0.2, 0.4, 0.8]])
    np.testing.assert_allclose(fuse_state_evidence(evidence), [2.25, 0.55])


def test_r4_only_severity_ignores_other_state_evidence():
    rows = [
        {
            "subject_id": "s1",
            "session": "task",
            "window_index": index,
            "multiview_evidence": 1000.0 + index,
            "oriented_baseline_evidence": -1000.0 - index,
            "mamba_moments_evidence": float(index),
        }
        for index in range(11)
    ]
    feature = subject_r4_features(rows, {"s1": "task"})[0]
    assert feature["R4_winsorized_std"] == winsorized_std(range(11))
    assert feature["severity_feature"] == -winsorized_std(range(11))


def test_uniform_anchor_mask_selects_u3_to_u6():
    rows = [
        {"session": "rest", "window_index": index, "subject_id": "s1"}
        for index in range(16)
    ]
    mask = uniform_anchor_mask(rows, "rest")
    assert np.flatnonzero(mask).tolist() == [4, 6, 9, 11]


def test_state_logit_helper_returns_only_state_logits():
    embeddings = torch.tensor(
        [
            [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[-1.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0]],
            [[-1.0, 0.0], [-0.25, 0.0], [0.25, 0.0], [1.0, 0.0]],
        ]
    )
    labels = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 3)
    logits = leave_one_subject_out_logits(embeddings, labels, torch.tensor(2.0))
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == labels.shape


def test_legacy_a1_checkpoint_keys_are_migrated_explicitly():
    model = torch.nn.Linear(2, 1)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    state["mamba_scale"] = torch.tensor(1e-3)
    state["gated_kan.alpha"] = torch.tensor(0.0)
    ignored = load_checkpoint_state_dict(model, state, source="test")
    assert ignored == ("gated_kan.alpha", "mamba_scale")
