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

        self.collision_request = hppfcl.CollisionRequest()
        self.collision_geometries = [
            geometry_object.geometry for geometry_object in self.collision_model.geometryObjects
        ]
        self.bounding_radii = np.array(
            [geometry.aabb_radius for geometry in self.collision_geometries],
            dtype=np.float64,
        )
        self.local_aabb_centers = np.array(
            [
                (geometry.aabb_local.min_ + geometry.aabb_local.max_) / 2.0
                for geometry in self.collision_geometries
            ],
            dtype=np.float64,
        )
        self._cached_box_signature = None
        self._cached_boxes = []
        self._cached_box_transforms = []
    
    def _update_geometry(self, q):
        q_full = self.to_configuration(q)

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q_full
        )

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

    def collides_with_boxes(self, q, box_centers, box_halfs):
        self._update_geometry(q)
        self._update_box_cache(box_centers, box_halfs)

        for geometry, placement, robot_radius, local_center in zip(
            self.collision_geometries,
            self.collision_data.oMg,
            self.bounding_radii,
            self.local_aabb_centers,
        ):
            world_aabb_center = placement.rotation @ local_center + placement.translation
            robot_transform = hppfcl.Transform3f(placement.rotation, placement.translation)

            candidate_mask = (
                np.linalg.norm(self._cached_box_centers - world_aabb_center, axis=1)
                <= (robot_radius + self._cached_box_bounding_radii)
            )
            if not np.any(candidate_mask):
                continue

            for idx in np.where(candidate_mask)[0]:
                result = hppfcl.CollisionResult()

                hppfcl.collide(
                    geometry,
                    robot_transform,
                    self._cached_boxes[idx],
                    self._cached_box_transforms[idx],
                    self.collision_request,
                    result
                )

                if result.isCollision():
                    return True

        return False

    def _update_box_cache(self, box_centers, box_halfs):
        centers = np.asarray(box_centers, dtype=np.float64)
        half_extents = np.asarray(box_halfs, dtype=np.float64)
        signature = (centers.shape, half_extents.shape, centers.tobytes(), half_extents.tobytes())

        if signature == self._cached_box_signature:
            return

        self._cached_box_signature = signature
        self._cached_box_centers = centers
        self._cached_box_half_extents = half_extents
        self._cached_box_bounding_radii = np.linalg.norm(half_extents, axis=1)
        self._cached_boxes = [hppfcl.Box(*(2.0 * half)) for half in half_extents]
        self._cached_box_transforms = [hppfcl.Transform3f(np.eye(3), center) for center in centers]
