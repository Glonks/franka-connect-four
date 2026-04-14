import numpy as np
import pinocchio as pin
import RobotUtil as rt

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

        self.collision_boxes = []
        for geometry_index, geometry_object in enumerate(self.collision_model.geometryObjects):
            geometry_object.geometry.computeLocalAABB()

            aabb = geometry_object.geometry.aabb_local
            dimensions = np.asarray(aabb.max_ - aabb.min_, dtype=float)
            if np.any(dimensions <= 0.0):
                continue

            local_center = np.asarray(0.5 * (aabb.min_ + aabb.max_), dtype=float)
            self.collision_boxes.append((geometry_index, local_center, dimensions))

    def clip(self, q):
        q = np.asarray(q, dtype=float)

        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def neutral(self):
        return np.zeros(7, dtype=float)

    def to_configuration(self, q, gripper_q=None):
        q_full = pin.neutral(self.model)
        q_full[:7] = self.clip(q)

        if q_full.shape[0] > 7:
            if gripper_q is None:
                q_full[7:] = 0.0
            else:
                q_full[7:] = np.clip(
                    np.asarray(gripper_q, dtype=float),
                    self.model.lowerPositionLimit[7:],
                    self.model.upperPositionLimit[7:],
                )

        return q_full

    def forward_kinematics(self, q, frame=None):
        frame_id = self.ee_frame_id if frame is None else self.model.getFrameId(frame)
        q_full = self.to_configuration(q)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        placement = self.data.oMf[frame_id]

        orientation = R.from_matrix(placement.rotation).as_quat(scalar_first=True)

        return Pose(position=placement.translation.copy(), orientation=orientation)

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

    def collision_box_descriptors(self, q, gripper_q=None):
        q_full = self.to_configuration(q, gripper_q=gripper_q)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q_full,
        )

        descriptors = []
        for geometry_index, local_center, dimensions in self.collision_boxes:
            placement = self.collision_data.oMg[geometry_index]

            H = np.eye(4)
            H[:3, :3] = placement.rotation
            H[:3, 3] = placement.rotation @ local_center + placement.translation

            descriptors.append(rt.BlockDesc2Points(H, dimensions))

        return descriptors
