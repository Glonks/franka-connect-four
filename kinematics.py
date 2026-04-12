import numpy as np
import pinocchio as pin

from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


LINK_SPHERES = {
    "link1": 0.08,
    "link2": 0.08,
    "link3": 0.08,
    "link4": 0.08,
    "link5": 0.07,
    "link6": 0.06,
    "link7": 0.05,
    "hand": 0.06,
}


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    orientation: np.ndarray


@dataclass(frozen=True)
class CollisionSphere:
    frame: str
    radius: float


class PandaKinematics:
    def __init__(self, mjcf_path, ee_frame="hand", collision_spheres=None):
        self.mjcf_path = mjcf_path
        self.model = pin.buildModelFromMJCF(mjcf_path)
        self.data = self.model.createData()

        self.arm_joint_ids = [self.model.getJointId(f"joint{i}") for i in range(1, 8)]
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        self.ee_frame = ee_frame

        self.joint_limits = np.column_stack([
            self.model.lowerPositionLimit[:7],
            self.model.upperPositionLimit[:7],
        ])

        spheres = collision_spheres if collision_spheres is not None else LINK_SPHERES
        self.collision_spheres = [CollisionSphere(name, radius) for name, radius in spheres.items()]

    def clip(self, q):
        q = np.asarray(q, dtype=float)

        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def neutral(self):
        return np.zeros(7, dtype=float)

    def to_configuration(self, q):
        q_full = pin.neutral(self.model)
        q_full[:7] = self.clip(q)

        if q_full.shape[0] > 7:
            q_full[7:] = 0.0

        return q_full

    def forward_kinematics(self, q, frame=None):
        frame_id = self.ee_frame_id if frame is None else self.model.getFrameId(frame)
        q_full = self.to_configuration(q)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        placement = self.data.oMf[frame_id]

        # TODO: collapse this
        quat_xyzw = R.from_matrix(placement.rotation).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return Pose(position=placement.translation.copy(), orientation=quat_wxyz)

    def jacobian(self, q, frame=None):
        frame_id = self.ee_frame_id if frame is None else self.model.getFrameId(frame)

        q_full = self.to_configuration(q)

        pin.computeJointJacobians(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            q_full,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        return J[:, :7].copy()

    def collision_sphere_centers(self, q):
        q_full = self.to_configuration(q)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        centers = []
        for sphere in self.collision_spheres:
            frame_id = self.model.getFrameId(sphere.frame)
            placement = self.data.oMf[frame_id]

            centers.append((sphere.frame, placement.translation.copy(), sphere.radius))

        return centers
