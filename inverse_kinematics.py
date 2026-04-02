import numpy as np
import mujoco as mj
import copy

from scipy.spatial.transform import Rotation as R

from actions import CommonPoses


class IKSolver:
    def __init__(
        self,
        model,
        body_name="hand",
        max_iterations=1000,
        position_tolerance=1e-3,
        rotation_tolerance=1e-3,
        step_size=0.5,
        W=None,
        C=None,
        alpha=0.1
    ):
        W = W if W is not None else np.eye(7)
        C = C if C is not None else np.eye(6)

        self.model = model
        self.body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.step_size = step_size
        self.W_inv = np.linalg.inv(W)
        self.C_inv = np.linalg.inv(C)
        self.alpha = alpha

        self.joint_limits = self.model.jnt_range[:7].T

    def solve(self, data_original, target_pose, bias=None):
        target_position, target_orientation = target_pose
        target_orientation = R.from_quat(target_orientation, scalar_first=True)

        data = copy.deepcopy(data_original)
        nv = self.model.nv

        for _ in range(self.max_iterations):
            mj.mj_forward(self.model, data)

            current_position = data.xpos[self.body_id].copy()
            position_error = target_position - current_position

            current_orientation = R.from_quat(data.xquat[self.body_id].copy(), scalar_first=True)
            orientation_error = (target_orientation * current_orientation.inv()).as_rotvec()

            position_error_norm = np.linalg.norm(position_error)
            orientation_error_norm = np.linalg.norm(orientation_error)

            error = np.concatenate([
                position_error,
                orientation_error
            ])

            converged = (
                position_error_norm < self.position_tolerance and
                orientation_error_norm < self.rotation_tolerance
            )
            if converged:
                return data.qpos[:7].copy(), True, error
        
            J_position = np.zeros((3, nv))
            J_orientation = np.zeros((3, nv))
            mj.mj_jacBody(
                self.model,
                data,
                J_position,
                J_orientation,
                self.body_id
            )
            J = np.vstack([J_position, J_orientation])[:, :7]

            J_prime = self.W_inv @ J.T @ np.linalg.inv(J @ self.W_inv @ J.T + self.C_inv)
            dq = J_prime @ error

            if bias is not None:
                J_null = np.eye(7) - J_prime @ J
                dq += self.alpha * (J_null @ (bias - data.qpos[:7]))

            data.qpos[:7] += self.step_size * dq
            data.qpos[:7] = np.clip(
                data.qpos[:7],
                self.joint_limits[0],
                self.joint_limits[1]
            )

        return data.qpos[:7].copy(), False, error
