from pathlib import Path

import numpy as np
import pytest

from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig


def _build_five_finger_optimizer():
    robot_dir = Path(__file__).parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    config_path = get_default_config_path(
        RobotName.shadow, RetargetingType.dexpilot, HandType.right
    )
    config = RetargetingConfig.load_from_file(
        config_path,
        {"low_pass_alpha": 0, "normal_delta": 0, "scaling_factor": 1.0},
    )
    return config.build().optimizer


def _objective_inputs(optimizer):
    target = np.full((len(optimizer.origin_link_names), 3), 0.06)
    fixed = optimizer.robot.q0[optimizer.idx_pin2fixed]
    qpos = optimizer.robot.q0[optimizer.idx_pin2target]
    return target, fixed, qpos


def _configure_fist_links(optimizer):
    optimizer.set_calibration_fist_links(
        ["thbase", "ffknuckle", "mfknuckle", "rfknuckle", "lfmetacarpal"]
    )


def _tight_joint_targets(values=None):
    values = values or {}
    return {
        "index": {"FFJ3": values.get("FFJ3", 0.0)},
        "middle": {"MFJ3": values.get("MFJ3", 0.0)},
        "ring": {"RFJ3": values.get("RFJ3", 0.0)},
        "pinky": {"LFJ3": values.get("LFJ3", 0.0)},
    }


def test_zero_calibration_weight_preserves_objective_and_gradient():
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    objective = optimizer.get_objective_function(target, fixed, qpos)
    baseline_grad = np.empty_like(qpos)
    baseline_loss = objective(qpos, baseline_grad)

    optimizer.set_calibration_pinch_targets(
        np.full((4, 3), 0.01), np.zeros(4), np.ones(4)
    )
    objective = optimizer.get_objective_function(target, fixed, qpos)
    zero_grad = np.empty_like(qpos)
    zero_loss = objective(qpos, zero_grad)

    assert zero_loss == baseline_loss
    np.testing.assert_array_equal(zero_grad, baseline_grad)


@pytest.mark.parametrize("mask", [None, [True, False, True, False], [False] * 4])
def test_calibration_pair_gradient_matches_finite_difference(mask):
    optimizer = _build_five_finger_optimizer()
    if mask is not None:
        optimizer.set_calibration_pinch_mask(np.asarray(mask))
    target, fixed, qpos = _objective_inputs(optimizer)
    optimizer.set_calibration_pinch_targets(
        np.array(
            [
                [0.01, 0.00, 0.00],
                [0.00, 0.01, 0.00],
                [0.00, 0.00, 0.01],
                [0.01, 0.01, 0.00],
            ]
        ),
        np.array([1.0, 2.0, 4.0, 8.0]),
        np.array([1.0, 0.8, 0.6, 0.4]),
    )
    objective = optimizer.get_objective_function(target, fixed, qpos)
    analytical = np.empty_like(qpos)
    objective(qpos, analytical)

    epsilon = 1e-6
    finite_difference = np.empty_like(qpos)
    for index in range(len(qpos)):
        step = np.zeros_like(qpos)
        step[index] = epsilon
        finite_difference[index] = (
            objective(qpos + step, np.array([]))
            - objective(qpos - step, np.array([]))
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(analytical, finite_difference, rtol=2e-3, atol=2e-4)


def test_pinch_mask_disables_native_projection_and_derived_finger_pairs():
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    target[:10] = 0.001
    optimizer.get_objective_function(target, fixed, qpos)
    assert optimizer.projected.all()
    optimizer.set_calibration_pinch_mask(np.asarray([True, False, False, False]))
    optimizer.set_calibration_pinch_targets(np.zeros((4, 3)), np.ones(4), np.ones(4))
    optimizer.get_objective_function(target, fixed, qpos)
    np.testing.assert_array_equal(optimizer.projected, [True] + [False] * 9)
    np.testing.assert_array_equal(optimizer.calibration_pair_coefficients, [1, 0, 0, 0])
    optimizer.set_calibration_pinch_targets(np.zeros((4, 3)), np.zeros(4), np.ones(4))
    optimizer.get_objective_function(target, fixed, qpos)
    assert optimizer.projected[0]
    assert optimizer.calibration_pair_targets is None


def test_all_pinches_disabled_fall_back_to_ordinary_native_targets():
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    target[:10] = 0.001
    optimizer.set_calibration_pinch_mask(np.zeros(4, dtype=bool))
    objective = optimizer.get_objective_function(target, fixed, qpos)
    masked_grad = np.empty_like(qpos)
    masked_loss = objective(qpos, masked_grad)
    assert not optimizer.projected.any()
    optimizer.set_calibration_pinch_mask(np.ones(4, dtype=bool))
    optimizer.project_dist = optimizer.escape_dist = 0.0
    objective = optimizer.get_objective_function(target, fixed, qpos)
    normal_grad = np.empty_like(qpos)
    assert objective(qpos, normal_grad) == masked_loss
    np.testing.assert_array_equal(normal_grad, masked_grad)


def test_explicit_all_enabled_pinch_mask_preserves_legacy_objective():
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    target[:10] = 0.001
    objective = optimizer.get_objective_function(target, fixed, qpos)
    baseline_grad = np.empty_like(qpos)
    baseline_loss = objective(qpos, baseline_grad)
    optimizer.set_calibration_pinch_mask(np.ones(4, dtype=bool))
    objective = optimizer.get_objective_function(target, fixed, qpos)
    explicit_grad = np.empty_like(qpos)
    assert objective(qpos, explicit_grad) == baseline_loss
    np.testing.assert_array_equal(explicit_grad, baseline_grad)


@pytest.mark.parametrize("mask", [[True], [1, 0, 0, 0], None])
def test_invalid_pinch_mask_is_rejected(mask):
    optimizer = _build_five_finger_optimizer()
    with pytest.raises(ValueError, match="Pinch mask"):
        optimizer.set_calibration_pinch_mask(mask)


def test_zero_fist_override_preserves_objective_and_gradient():
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    objective = optimizer.get_objective_function(target, fixed, qpos)
    baseline_grad = np.empty_like(qpos)
    baseline_loss = objective(qpos, baseline_grad)

    _configure_fist_links(optimizer)
    optimizer.set_calibration_fist_targets(
        np.full((5, 3), 0.01), np.zeros(5), 0.0
    )
    objective = optimizer.get_objective_function(target, fixed, qpos)
    zero_grad = np.empty_like(qpos)
    zero_loss = objective(qpos, zero_grad)

    assert zero_loss == baseline_loss
    np.testing.assert_array_equal(zero_grad, baseline_grad)


@pytest.mark.parametrize("finger_activations", [None, [0.0, 0.4, 0.7, 1.0]])
def test_calibration_fist_gradient_matches_finite_difference(finger_activations):
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    _configure_fist_links(optimizer)
    optimizer.set_calibration_fist_targets(
        np.array(
            [
                [0.01, 0.00, 0.00],
                [0.00, 0.01, 0.00],
                [0.00, 0.00, 0.01],
                [0.01, 0.01, 0.00],
                [0.00, 0.01, 0.01],
            ]
        ),
        np.array([0.0, 2.0, 4.0, 8.0, 16.0]),
        0.8,
        finger_activations=finger_activations,
    )
    objective = optimizer.get_objective_function(target, fixed, qpos)
    analytical = np.empty_like(qpos)
    objective(qpos, analytical)

    epsilon = 1e-6
    finite_difference = np.empty_like(qpos)
    for index in range(len(qpos)):
        step = np.zeros_like(qpos)
        step[index] = epsilon
        finite_difference[index] = (
            objective(qpos + step, np.array([]))
            - objective(qpos - step, np.array([]))
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(analytical, finite_difference, rtol=2e-3, atol=2e-4)


def test_thumb_fist_weight_does_not_change_objective_or_gradient():
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    _configure_fist_links(optimizer)
    fist_targets = np.full((5, 3), 0.01)
    optimizer.set_calibration_fist_targets(fist_targets, np.zeros(5), 1.0)
    objective = optimizer.get_objective_function(target, fixed, qpos)
    baseline_grad = np.empty_like(qpos)
    baseline_loss = objective(qpos, baseline_grad)

    optimizer.set_calibration_fist_targets(
        fist_targets, np.array([64.0, 0.0, 0.0, 0.0, 0.0]), 1.0
    )
    objective = optimizer.get_objective_function(target, fixed, qpos)
    thumb_grad = np.empty_like(qpos)
    thumb_loss = objective(qpos, thumb_grad)

    assert optimizer.calibration_fist_targets is None
    assert thumb_loss == baseline_loss
    np.testing.assert_array_equal(thumb_grad, baseline_grad)


def test_full_fist_activation_disables_calibrated_pinch_pair():
    optimizer = _build_five_finger_optimizer()
    _configure_fist_links(optimizer)
    optimizer.set_calibration_fist_targets(np.zeros((5, 3)), np.zeros(5), 1.0)
    optimizer.set_calibration_pinch_targets(
        np.zeros((4, 3)), np.ones(4), np.ones(4)
    )

    assert optimizer.calibration_pair_targets is None
    assert optimizer.calibration_pair_coefficients is None


@pytest.mark.parametrize("finger_activations", [None, [0.0, 0.4, 0.7, 1.0]])
def test_tight_joint_gradient_matches_finite_difference(finger_activations):
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    _configure_fist_links(optimizer)
    optimizer.set_calibration_fist_targets(
        np.zeros((5, 3)), np.zeros(5), 0.8, finger_activations=finger_activations
    )
    optimizer.set_calibration_tight_joint_targets(
        _tight_joint_targets(
            {
                name: float(qpos[optimizer.target_joint_names.index(name)])
                for name in ("FFJ3", "MFJ3", "RFJ3", "LFJ3")
            }
            | {"FFJ3": float(qpos[optimizer.target_joint_names.index("FFJ3")] + 0.4)}
        )
    )
    objective = optimizer.get_objective_function(target, fixed, qpos)
    analytical = np.empty_like(qpos)
    objective(qpos, analytical)

    epsilon = 1e-6
    numerical = np.empty_like(qpos)
    for index in range(len(qpos)):
        step = np.zeros_like(qpos)
        step[index] = epsilon
        numerical[index] = (
            objective(qpos + step, np.array([]))
            - objective(qpos - step, np.array([]))
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(analytical, numerical, rtol=2e-3, atol=2e-4)


def test_tight_joint_target_requires_fist_activation():
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    baseline = optimizer.get_objective_function(target, fixed, qpos)
    baseline_grad = np.empty_like(qpos)
    baseline_loss = baseline(qpos, baseline_grad)
    optimizer.set_calibration_tight_joint_targets(
        _tight_joint_targets({"FFJ3": 1.0})
    )
    inactive = optimizer.get_objective_function(target, fixed, qpos)
    inactive_grad = np.empty_like(qpos)

    assert inactive(qpos, inactive_grad) == baseline_loss
    np.testing.assert_array_equal(inactive_grad, baseline_grad)


def test_extended_finger_ignores_tight_target_and_keeps_its_pinch_objective():
    optimizer = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(optimizer)
    _configure_fist_links(optimizer)
    optimizer.set_calibration_fist_targets(
        np.zeros((5, 3)), np.ones(5), 1.0,
        finger_activations=np.array([0.0, 1.0, 1.0, 1.0]),
    )
    np.testing.assert_array_equal(optimizer.calibration_fist_coefficients, [0, 0, 1, 1, 1])
    optimizer.set_calibration_pinch_targets(np.zeros((4, 3)), np.ones(4), np.ones(4))
    np.testing.assert_array_equal(optimizer.calibration_pair_coefficients, [1, 0, 0, 0])
    optimizer.set_calibration_tight_joint_targets(_tight_joint_targets())
    objective = optimizer.get_objective_function(target, fixed, qpos)
    baseline_grad = np.empty_like(qpos)
    baseline_loss = objective(qpos, baseline_grad)

    optimizer.set_calibration_tight_joint_targets(_tight_joint_targets({"FFJ3": 1.5}))
    objective = optimizer.get_objective_function(target, fixed, qpos)
    changed_grad = np.empty_like(qpos)
    assert objective(qpos, changed_grad) == baseline_loss
    np.testing.assert_array_equal(changed_grad, baseline_grad)


@pytest.mark.parametrize("amounts", [[0.0], [np.nan] * 4, [-0.1] * 4, [1.1] * 4])
def test_invalid_finger_fist_activations_are_rejected(amounts):
    optimizer = _build_five_finger_optimizer()
    _configure_fist_links(optimizer)
    with pytest.raises(ValueError, match="Finger fist activations"):
        optimizer.set_calibration_fist_targets(
            np.zeros((5, 3)), np.ones(5), 1.0, finger_activations=amounts
        )


@pytest.mark.parametrize(("joint", "value"), (("missing_joint", 0.0), ("FFJ3", np.nan)))
def test_tight_joint_targets_require_known_finite_sources(joint, value):
    optimizer = _build_five_finger_optimizer()
    targets = _tight_joint_targets()
    targets["index"] = {joint: value}

    with pytest.raises(ValueError, match="optimized joints|finite"):
        optimizer.set_calibration_tight_joint_targets(targets)


def test_full_fist_solution_is_pulled_toward_tight_joint_target():
    baseline = _build_five_finger_optimizer()
    assisted = _build_five_finger_optimizer()
    target, fixed, qpos = _objective_inputs(baseline)
    _configure_fist_links(baseline)
    _configure_fist_links(assisted)
    for optimizer in (baseline, assisted):
        optimizer.set_calibration_fist_targets(
            np.zeros((5, 3)), np.zeros(5), 1.0
        )
    tight_target = 1.2
    assisted.set_calibration_tight_joint_targets(
        _tight_joint_targets(
            {
                name: float(qpos[assisted.target_joint_names.index(name)])
                for name in ("FFJ3", "MFJ3", "RFJ3", "LFJ3")
            }
            | {"FFJ3": tight_target}
        )
    )

    baseline_qpos = baseline.retarget(target, fixed, qpos)
    assisted_qpos = assisted.retarget(target, fixed, qpos)
    index = assisted.target_joint_names.index("FFJ3")

    assert abs(assisted_qpos[index] - tight_target) < 0.75 * abs(
        baseline_qpos[index] - tight_target
    )
