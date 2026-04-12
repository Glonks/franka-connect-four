import numpy as np

from scipy.spatial.transform import Rotation as R


class IKSolver:
    def __init__(
        self,
        robot_model,
        frame_name="hand",
        max_iterations=1000,
        position_tolerance=1e-3,
        rotation_tolerance=1e-3,
        step_size=0.5,
        W=None,
        C=None,
        alpha=0.1,
    ):
        W = W if W is not None else np.eye(7)
        C = C if C is not None else np.eye(6)

        self.robot_model = robot_model
        self.frame_name = frame_name
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.step_size = step_size
        self.W_inv = np.linalg.inv(W)
        self.C_inv = np.linalg.inv(C)
        self.alpha = alpha
        self.joint_limits = self.robot_model.joint_limits

    def solve(self, q_init, target_pose, bias=None):
        target_position, target_orientation = target_pose
        target_position = np.asarray(target_position, dtype=float)
        target_orientation = R.from_quat(np.asarray(target_orientation, dtype=float), scalar_first=True)

        q = self.robot_model.clip(q_init)

        for _ in range(self.max_iterations):
            current_pose = self.robot_model.forward_kinematics(q, frame=self.frame_name)
            current_orientation = R.from_quat(current_pose.orientation, scalar_first=True)

            position_error = target_position - current_pose.position
            orientation_error = (target_orientation * current_orientation.inv()).as_rotvec()
            error = np.concatenate([position_error, orientation_error])

            if (
                np.linalg.norm(position_error) < self.position_tolerance
                and np.linalg.norm(orientation_error) < self.rotation_tolerance
            ):
                return q.copy(), True, error

            J = self.robot_model.jacobian(q, frame=self.frame_name)
            J_prime = self.W_inv @ J.T @ np.linalg.inv(J @ self.W_inv @ J.T + self.C_inv)
            dq = J_prime @ error

            if bias is not None:
                J_null = np.eye(7) - J_prime @ J
                dq += self.alpha * (J_null @ (np.asarray(bias, dtype=np.float32) - q))

            q = self.robot_model.clip(q + self.step_size * dq)

        return q.copy(), False, error
