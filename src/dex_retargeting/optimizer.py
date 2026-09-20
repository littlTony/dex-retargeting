from abc import abstractmethod
from typing import List, Mapping, Optional

import nlopt
import numpy as np
import torch

from dex_retargeting.kinematics_adaptor import (
    KinematicAdaptor,
    MimicJointKinematicAdaptor,
)
from dex_retargeting.robot_wrapper import RobotWrapper


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        self.robot = robot
        self.num_joints = robot.dof

        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    f"Joint {target_joint_name} given does not appear to be in robot XML."
                )
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        self.idx_pin2fixed = np.array(
            [i for i in range(robot.dof) if i not in idx_pin2target], dtype=int
        )
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Target
        self.target_link_human_indices = target_link_human_indices

        # Free joint
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        # Kinematics adaptor
        self.adaptor: Optional[KinematicAdaptor] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}"
            )
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        if isinstance(adaptor, MimicJointKinematicAdaptor):
            fixed_idx = self.idx_pin2fixed
            mimic_idx = adaptor.idx_pin2mimic
            new_fixed_id = np.array(
                [x for x in fixed_idx if x not in mimic_idx], dtype=int
            )
            self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        """
        Compute the retargeting results using non-linear optimization
        Args:
            ref_value: the reference value in cartesian space as input, different optimizer has different reference
            fixed_qpos: the fixed value (not optimized) in retargeting, consistent with self.fixed_joint_names
            last_qpos: the last retargeting results or initial value, consistent with function return

        Returns: joint position of robot, the joint order and dim is consistent with self.target_joint_names

        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        objective_fn = self.get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(e)
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        pass

    @property
    def fixed_joint_names(self):
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]


class PositionOptimizer(Optimizer):
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses], axis=0
            )  # (n ,3)

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Cache link indices that will involve in kinematics computation
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class DexPilotOptimizer(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """

    retargeting_type = "DEXPILOT"
    # Soft-pose objective coefficient in loss units per radian squared.
    fist_tight_q_weight = 0.05

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        # gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)

        origin_link_index, task_link_index = self.generate_link_indices(
            self.num_fingers
        )

        if target_link_human_indices is None:
            target_link_human_indices = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.finger_tip_link_names = list(finger_tip_link_names)
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Sanity check and cache link indices
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self.set_dexpilot_cache(self.num_fingers, eta1, eta2)
        self.calibration_pair_targets = None
        self.calibration_pair_coefficients = None
        self.calibration_pinch_mask = np.ones(self.num_fingers - 1, dtype=bool)
        self.calibration_fist_targets = None
        self.calibration_fist_coefficients = None
        self.calibration_fist_activation = 0.0
        self.calibration_fist_finger_activations = np.zeros(self.num_fingers - 1)
        self.calibration_fist_tip_indices = None
        self.calibration_fist_proximal_indices = None
        self.calibration_tight_joint_indices = None
        self.calibration_tight_joint_targets = None
        self.calibration_tight_joint_fingers = None

    def set_calibration_fist_links(self, proximal_link_names: List[str]):
        """Add the five proximal links needed by the optional fist objective."""
        if self.num_fingers != 5 or len(proximal_link_names) != 5:
            raise ValueError("Calibrated fist targets require five proximal links")
        if len(set(proximal_link_names)) != 5:
            raise ValueError("Calibrated fist proximal links must be independent")
        for name in proximal_link_names:
            self.robot.get_link_index(name)
            if name not in self.computed_link_names:
                self.computed_link_names.append(name)
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)
        self.calibration_fist_tip_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in self.finger_tip_link_names]
        )
        self.calibration_fist_proximal_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in proximal_link_names]
        )

    def set_calibration_fist_targets(
        self,
        targets: np.ndarray,
        weights: np.ndarray,
        activation: float,
        finger_activations: Optional[np.ndarray] = None,
    ):
        """Set fist targets, capped by each finger's own curl when provided."""
        targets = np.asarray(targets, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        activation = float(activation)
        if targets.shape != (5, 3) or weights.shape != (5,):
            raise ValueError("Calibrated fist targets/weights must have shapes (5, 3)/(5,)")
        if not np.isfinite(targets).all() or not np.isfinite(weights).all():
            raise ValueError("Calibrated fist targets must be finite")
        if np.any(weights < 0.0) or not 0.0 <= activation <= 1.0:
            raise ValueError("Calibrated fist weights/activation are out of range")
        if self.calibration_fist_tip_indices is None:
            raise ValueError("Calibrated fist links must be configured before targets")
        amounts = np.full(4, activation)
        if finger_activations is not None:
            finger_activations = np.asarray(finger_activations, dtype=np.float64)
            if (
                finger_activations.shape != (4,)
                or not np.isfinite(finger_activations).all()
                or np.any((finger_activations < 0.0) | (finger_activations > 1.0))
            ):
                raise ValueError("Finger fist activations must be finite (4,) values in [0, 1]")
            amounts = np.minimum(amounts, finger_activations)
        self.calibration_fist_finger_activations = amounts
        coefficients = weights * np.concatenate(([0.0], amounts))
        # Keep the thumb under the native retargeting objective during fist assistance.
        coefficients[0] = 0.0
        self.calibration_fist_activation = activation
        self.calibration_fist_targets = targets if np.any(coefficients) else None
        self.calibration_fist_coefficients = (
            coefficients if self.calibration_fist_targets is not None else None
        )

    def set_calibration_tight_joint_targets(
        self, joint_positions: Mapping[str, Mapping[str, float]]
    ):
        """Set collision-solved four-finger source-joint targets in radians."""
        fingers = ("index", "middle", "ring", "pinky")
        if not isinstance(joint_positions, Mapping) or set(joint_positions) != set(
            fingers
        ):
            raise ValueError("Tight joint targets must define four non-thumb fingers")
        if any(
            not isinstance(joint_positions[finger], Mapping)
            or not joint_positions[finger]
            for finger in fingers
        ):
            raise ValueError("Tight source joints must be non-empty and independent")
        flat = [
            (name, value)
            for finger in fingers
            for name, value in joint_positions[finger].items()
        ]
        if len({name for name, _ in flat}) != len(flat):
            raise ValueError("Tight source joints must be non-empty and independent")
        unknown = {name for name, _ in flat} - set(self.target_joint_names)
        if unknown:
            raise ValueError(f"Tight joint targets are not optimized joints: {sorted(unknown)}")
        values = np.asarray([value for _, value in flat], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Tight joint targets must be finite")
        self.calibration_tight_joint_indices = np.asarray(
            [self.target_joint_names.index(name) for name, _ in flat], dtype=int
        )
        self.calibration_tight_joint_targets = values
        self.calibration_tight_joint_fingers = np.asarray(
            [index for index, finger in enumerate(fingers) for _ in joint_positions[finger]],
            dtype=int,
        )

    def set_calibration_pinch_mask(self, enabled: np.ndarray):
        """Gate both native projections and the optional calibrated pair loss."""
        enabled = np.asarray(enabled)
        if enabled.shape != (self.num_fingers - 1,) or enabled.dtype != np.bool_:
            raise ValueError("Pinch mask must contain one boolean per non-thumb finger")
        self.calibration_pinch_mask = enabled.copy()
        self.calibration_pair_targets = None
        self.calibration_pair_coefficients = None

    def set_calibration_pinch_targets(
        self,
        targets: np.ndarray,
        weights: np.ndarray,
        activations: np.ndarray,
    ):
        """Set optional thumb-to-finger targets for the next optimization."""
        targets = np.asarray(targets, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        activations = np.asarray(activations, dtype=np.float64)
        if self.num_fingers != 5 or targets.shape != (4, 3):
            raise ValueError("Calibrated pinch targets require a five-finger (4, 3) target")
        if weights.shape != (4,) or activations.shape != (4,):
            raise ValueError("Calibrated pinch weights and activations must have shape (4,)")
        if not all(np.isfinite(value).all() for value in (targets, weights, activations)):
            raise ValueError("Calibrated pinch targets must be finite")
        if np.any(weights < 0.0) or np.any((activations < 0.0) | (activations > 1.0)):
            raise ValueError("Calibrated pinch weights/activations are out of range")

        coefficients = (
            weights * activations * self.calibration_pinch_mask
            * (1.0 - self.calibration_fist_finger_activations)
        )
        self.calibration_pair_targets = targets if np.any(coefficients) else None
        self.calibration_pair_coefficients = (
            coefficients if self.calibration_pair_targets is not None else None
        )

    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array(
            [eta1] * (num_fingers - 1)
            + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2)
        )

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[:len_s1] &= self.calibration_pinch_mask
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)
        fist_activation = self.calibration_fist_activation
        if fist_activation > 0.0:
            amounts = self.calibration_fist_finger_activations
            projection_activations = np.concatenate(
                (
                    amounts,
                    np.maximum(
                        amounts[self.s2_project_index_origin],
                        amounts[self.s2_project_index_task],
                    ),
                )
            )
            weight = (
                (1.0 - projection_activations) * weight
                + projection_activations * normal_weight
            )

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                ]
            )
        )

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )  # (6, 3)
        if fist_activation > 0.0:
            reference_vec = (
                (1.0 - projection_activations[:, None]) * reference_vec
                + projection_activations[:, None] * normal_vec[:len_proj]
            )
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )  # (10, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)
        if self.calibration_pair_targets is None:
            torch_pair_targets = None
            torch_pair_coefficients = None
        else:
            torch_pair_targets = torch.as_tensor(self.calibration_pair_targets)
            torch_pair_coefficients = torch.as_tensor(
                self.calibration_pair_coefficients
            )
        if self.calibration_fist_targets is None:
            torch_fist_targets = None
            torch_fist_coefficients = None
        else:
            torch_fist_targets = torch.as_tensor(self.calibration_fist_targets)
            torch_fist_coefficients = torch.as_tensor(
                self.calibration_fist_coefficients
            )
        tight_q_active = (
            self.calibration_tight_joint_indices is not None
            and fist_activation > 0.0
        )
        if tight_q_active:
            tight_coefficients = (
                self.fist_tight_q_weight
                * self.calibration_fist_finger_activations[self.calibration_tight_joint_fingers]
            )

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
                * weight
                / (robot_vec.shape[0])
            ).sum()
            if torch_pair_targets is not None:
                robot_pair_vec = -robot_vec[:4]
                pair_dist = torch.norm(
                    robot_pair_vec - torch_pair_targets, dim=1, keepdim=False
                )
                huber_distance += (
                    self.huber_loss(pair_dist, torch.zeros_like(pair_dist))
                    * torch_pair_coefficients
                ).sum()
            if torch_fist_targets is not None:
                tip_pos = torch_body_pos[self.calibration_fist_tip_indices, :]
                proximal_pos = torch_body_pos[
                    self.calibration_fist_proximal_indices, :
                ]
                fist_dist = torch.norm(
                    tip_pos - proximal_pos - torch_fist_targets,
                    dim=1,
                    keepdim=False,
                )
                huber_distance += (
                    self.huber_loss(fist_dist, torch.zeros_like(fist_dist))
                    * torch_fist_coefficients
                ).sum()
            huber_distance = huber_distance.sum()
            result = huber_distance.cpu().detach().item()
            if tight_q_active:
                tight_diff = (
                    x[self.calibration_tight_joint_indices]
                    - self.calibration_tight_joint_targets
                )
                result += float(np.sum(tight_coefficients * tight_diff ** 2))

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)
                if tight_q_active:
                    grad_qpos[self.calibration_tight_joint_indices] += (
                        2.0 * tight_coefficients * tight_diff
                    )

                grad[:] = grad_qpos[:]

            return result

        return objective
