import numpy as np
import RobotUtil as rt

from dataclasses import dataclass, field
from enum import Enum, auto


def _check_point_overlap(robot_points, obstacle_points, axis) -> bool:
    projected_robot_points = robot_points @ axis
    projected_obstacle_points = obstacle_points @ axis

    max_projected_robot_point = np.max(projected_robot_points)
    min_projected_robot_point = np.min(projected_robot_points)
    max_projected_obstacle_point = np.max(projected_obstacle_points)
    min_projected_obstacle_point = np.min(projected_obstacle_points)

    return (
        (max_projected_robot_point >= min_projected_obstacle_point) and
        (max_projected_obstacle_point >= min_projected_robot_point)
    )


def _check_box_box_collision(robot_descriptor, obstacle_descriptor) -> bool:
    robot_points, robot_axes = robot_descriptor
    obstacle_points, obstacle_axes = obstacle_descriptor

    for i in range(3):
        if not _check_point_overlap(robot_points, obstacle_points, robot_axes[i]):
            return False

    for j in range(3):
        if not _check_point_overlap(robot_points, obstacle_points, obstacle_axes[j]):
            return False

    for i in range(3):
        for j in range(3):
            axis = np.cross(robot_axes[i], obstacle_axes[j])

            if np.linalg.norm(axis) < 1e-10:
                continue

            if not _check_point_overlap(robot_points, obstacle_points, axis):
                return False

    return True


@dataclass
class Tree:
    buffer: np.ndarray
    parents: list = field(default_factory=list)
    size: int = 0

    def append(self, q: np.ndarray, parent_index: int):
        self.buffer[self.size] = q
        self.parents.append(parent_index)

        self.size += 1

    def view(self) -> np.ndarray:
        return self.buffer[:self.size]

    def extract_path(self) -> list:
        path = []
        index = self.size - 1

        while index != -1:
            path.append(self.buffer[index].copy())
            index = self.parents[index]

        path.reverse()

        return path


class ConnectStatus(Enum):
    REACHED = auto()
    ADVANCED = auto()
    TRAPPED = auto()


class RRTPlanner:
    """RRT-Connect"""
    def __init__(
        self,
        robot_model,
        obstacles,
        step_size=0.15,
        max_iterations=5000,
        goal_threshold=0.3,
        shortcut_attempts=100
    ):
        self.robot_model = robot_model
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        self.shortcut_attempts = shortcut_attempts
        self.joint_limits = self.robot_model.joint_limits.copy()

        self.obstacle_centers = np.array([o[1] for o in obstacles], dtype=float)
        self.obstacle_half_extents = np.array([o[2] for o in obstacles], dtype=float)
        self.obstacle_boxes = []
        for center, half_extents in zip(self.obstacle_centers, self.obstacle_half_extents):
            H = np.eye(4)
            H[:3, 3] = center

            self.obstacle_boxes.append(rt.BlockDesc2Points(H, 2.0 * half_extents))

    def _is_collision_free(self, q, gripper_q=None) -> bool:
        for robot_descriptor in self.robot_model.collision_box_descriptors(q, gripper_q=gripper_q):
            for obstacle_descriptor in self.obstacle_boxes:
                if _check_box_box_collision(robot_descriptor, obstacle_descriptor):
                    return False

        return True

    def _is_edge_free(self, q1, q2, gripper_q=None) -> bool:
        distance = np.linalg.norm(q2 - q1)
        steps = max(2, int(distance / self.step_size))

        for i in range(steps + 1):
            t = i / steps
            q = q1 + t * (q2 - q1)

            if not self._is_collision_free(q, gripper_q=gripper_q):
                return False

        return True

    def _sample(self) -> np.ndarray:
        return np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])

    def _nearest(self, tree, q) -> int:
        difference = tree.view() - q
        distances = np.sum(difference * difference, axis=1)

        return int(np.argmin(distances))

    def _steer(self, q_near, q_target) -> np.ndarray:
        difference = q_target - q_near
        distance = np.linalg.norm(difference)

        if distance <= self.step_size:
            return q_target.copy()

        return q_near + (difference / distance) * self.step_size

    def _extend(self, tree, q_target, gripper_q=None) -> ConnectStatus:
        nearest_index = self._nearest(tree, q_target)
        q_nearest = tree.buffer[nearest_index]

        q_new = self.robot_model.clip(self._steer(q_nearest, q_target))

        if not self._is_edge_free(q_nearest, q_new, gripper_q=gripper_q):
            return ConnectStatus.TRAPPED
 
        tree.append(q_new, nearest_index)
 
        if np.linalg.norm(q_new - q_target) < self.goal_threshold:
            return ConnectStatus.REACHED

        return ConnectStatus.ADVANCED

    def _connect(self, tree, q_target, gripper_q=None) -> ConnectStatus:
        status = ConnectStatus.ADVANCED

        while status == ConnectStatus.ADVANCED:
            status = self._extend(tree, q_target, gripper_q=gripper_q)

        return status

    def _stitch(self, tree_1, tree_2) -> list:
        path_1 = tree_1.extract_path()
        path_2 = tree_2.extract_path()

        path_2.reverse()

        return path_1 + path_2[1:]

    def _shortcut(self, path, gripper_q=None) -> list:
        for _ in range(self.shortcut_attempts):
            if len(path) <= 2:
                break

            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))

            if self._is_edge_free(path[i], path[j], gripper_q=gripper_q):
                path = path[: i + 1] + path[j:]

        return path

    def plan(self, q_start, q_goal, gripper_q=None):
        q_start = self.robot_model.clip(np.asarray(q_start, dtype=float))
        q_goal = self.robot_model.clip(np.asarray(q_goal, dtype=float))

        if not self._is_collision_free(q_start, gripper_q=gripper_q):
            print("Start configuration is in collision")
            return None

        if not self._is_collision_free(q_goal, gripper_q=gripper_q):
            print("Goal configuration is in collision")
            return None

        if self._is_edge_free(q_start, q_goal, gripper_q=gripper_q):
            return [q_start.copy(), q_goal.copy()]

        worst_case_size = self.max_iterations + 2

        tree_start = Tree(np.empty((worst_case_size, q_start.shape[0]), dtype=float))
        tree_goal = Tree(np.empty((worst_case_size, q_start.shape[0]), dtype=float))

        tree_start.append(q_start, -1)
        tree_goal.append(q_goal, -1)

        tree_1, tree_2 = tree_start, tree_goal

        for _ in range(self.max_iterations):
            q_sample = self._sample()

            if self._extend(tree_1, q_sample, gripper_q=gripper_q) == ConnectStatus.TRAPPED:
                tree_1, tree_2 = tree_2, tree_1
                continue

            q_new = tree_1.buffer[tree_1.size - 1]
            if self._connect(tree_2, q_new, gripper_q=gripper_q) == ConnectStatus.REACHED:
                path = self._stitch(tree_start, tree_goal)
                path = self._shortcut(path, gripper_q=gripper_q)

                return path

            tree_1, tree_2 = tree_2, tree_1

        return None
