import numpy as np
import pinocchio as pin
import hppfcl

from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    orientation: np.ndarray


class PandaKinematics:
    def __init__(self, mjcf_path, ee_frame="hand"):
        self.mjcf_path = mjcf_path
        self.model = pin.buildModelFromMJCF(mjcf_path)
        self.data = self.model.createData()
        self.collision_model = pin.buildGeomFromMJCF(
            self.model,
            mjcf_path,
            pin.GeometryType.COLLISION,
        )
        self.collision_data = pin.GeometryData(self.collision_model)

        self.arm_joint_ids = [self.model.getJointId(f"joint{i}") for i in range(1, 8)]
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        self.ee_frame = ee_frame

        self.joint_limits = np.column_stack([
            self.model.lowerPositionLimit[:7],
            self.model.upperPositionLimit[:7],
        ])

    def clip(self, q):
        q = np.asarray(q, dtype=np.float32)

        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def neutral(self):
        return np.zeros(7, dtype=np.float32)

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

    def _update_geometry(self, q):
        q_full = self.to_configuration(q)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q_full,
        )

    def collides_with_boxes(self, q, box_centers, box_halfs):
        self._update_geometry(q)

        request = hppfcl.CollisionRequest()
        for geometry_object, placement in zip(
            self.collision_model.geometryObjects,
            self.collision_data.oMg,
        ):
            robot_transform = hppfcl.Transform3f(placement.rotation, placement.translation)
            for center, half in zip(box_centers, box_halfs):
                box = hppfcl.Box(*(2.0 * np.asarray(half, dtype=np.float32)))
                box_transform = hppfcl.Transform3f(np.eye(3), np.asarray(center, dtype=np.float32))
                result = hppfcl.CollisionResult()
                hppfcl.collide(
                    geometry_object.geometry,
                    robot_transform,
                    box,
                    box_transform,
                    request,
                    result,
                )
                if result.isCollision():
                    return True

        return False
