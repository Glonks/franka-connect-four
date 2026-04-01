import numpy as np
import mujoco as mj
import copy


class IKSolver:
    def __init__(
        self,
        model,
        body_name="hand",
        max_iterations=1000,
        position_tolerance=1e-3,
        rotation_tolerance=1e-3,
        step_size=0.5,
        W = None,
        C = None
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

        self.joint_limits = self.model.jnt_range[:7].T

    def solve(self, data_original, target_pose):
        data = copy.deepcopy(data_original)
        nv = self.model.nv

        for _ in range(self.max_iterations):
            mj.mj_forward(self.model, data)

            target_position, target_orientation = target_pose

            current_position = data.xpos[self.body_id].copy()
            position_error = target_position - current_position

            current_orientation = data.xquat[self.body_id]
            orientation_error = np.zeros(3)
            # if np.dot(target_orientation, current_orientation) < 0:
            #     target_orientation = -target_orientation
            mj.mju_subQuat(orientation_error, target_orientation, current_orientation)

            position_error_norm = np.linalg.norm(position_error)
            orientation_error_norm = np.linalg.norm(orientation_error)

            position_error_scaled = position_error
            if position_error_norm > 0:
                position_error_scaled = (position_error / position_error_norm) * min(position_error_norm, self.step_size)

            orientation_error_scaled = orientation_error
            if orientation_error_norm > 0:
                axis = orientation_error / orientation_error_norm
                orientation_error_scaled = axis * min(orientation_error_norm, self.step_size)

            error = np.concatenate([
                position_error,
                orientation_error
            ])

            error_scaled = np.concatenate([
                position_error_scaled,
                orientation_error_scaled
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

            data.qpos[:7] += J_prime @ error_scaled
            data.qpos[:7] = np.clip(
                data.qpos[:7],
                self.joint_limits[0],
                self.joint_limits[1]
            )

        return data.qpos[:7].copy(), False, error
