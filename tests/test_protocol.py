from biofoundation_v1.protocol import EXPECTED, PROTOCOL


def test_locked_protocol():
    assert PROTOCOL["split_seed"] == 42
    assert PROTOCOL["training_seed"] == 1001
    assert PROTOCOL["severity_weight"] == 0.3
    assert PROTOCOL["outer_folds"] == 5
    assert PROTOCOL["inner_folds"] == 0
    assert PROTOCOL["state_fusion"] == "(R1 + R2 + 2R4) / 4"
    assert PROTOCOL["severity"] == "R4-only"
    assert set(EXPECTED) == {"monifeixing", "vrq", "city"}
