import numpy as np
import RobotUtil as rt

from dataclasses import dataclass


def _check_point_overlap(robot_points, obstacle_points, axis):
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


def _check_box_box_collision(robot_descriptor, obstacle_descriptor):
    # rp, ra = map(np.asarray, robot_descriptor)   # rp: (9,3), ra: (3,3)
    # op, oa = map(np.asarray, obstacle_descriptor)
    # cross_axes = np.cross(ra[:, None, :], oa[None, :, :]).reshape(-1, 3)
    # axes = np.vstack([ra, oa, cross_axes])                     # (15,3)
    # norms = np.linalg.norm(axes, axis=1)
    # axes = axes[norms > np.finfo(float).eps]                                   # drop degenerate axes
    # axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)  # normalize optional
    # # project all points onto all axes at once
    # r_proj = rp @ axes.T                                       # (9,k)
    # o_proj = op @ axes.T                                       # (9,k)
    # r_min, r_max = r_proj.min(axis=0), r_proj.max(axis=0)
    # o_min, o_max = o_proj.min(axis=0), o_proj.max(axis=0)
    # # overlap on every axis
    # return np.all((r_max >= o_min) & (o_max >= r_min))

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
    size: int = 0

    def append(self, x):
        self.buffer[self.size] = x
        self.size += 1

    def view(self):
        return self.buffer[:self.size]


class RRTPlanner:
    def __init__(
        self,
        robot_model,
        obstacles,
        step_size=0.15,
        max_iterations=5000,
        goal_bias=0.2,
        goal_threshold=0.3,
        shortcut_attempts=100,
        free_edge_checks=10
    ):
        self.robot_model = robot_model
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.goal_threshold = goal_threshold
        self.shortcut_attempts = shortcut_attempts
        self.free_edge_checks = free_edge_checks
        self.joint_limits = self.robot_model.joint_limits.copy()

        self.obstacle_centers = np.array([o[1] for o in obstacles], dtype=float)
        self.obstacle_half_extents = np.array([o[2] for o in obstacles], dtype=float)
        self.obstacle_boxes = []
        for center, half_extents in zip(self.obstacle_centers, self.obstacle_half_extents):
            H = np.eye(4)
            H[:3, 3] = center

            self.obstacle_boxes.append(rt.BlockDesc2Points(H, 2.0 * half_extents))

    def _is_collision_free(self, q, gripper_q=None):
        for robot_descriptor in self.robot_model.collision_box_descriptors(q, gripper_q=gripper_q):
            for obstacle_descriptor in self.obstacle_boxes:
                if _check_box_box_collision(robot_descriptor, obstacle_descriptor):
                    return False

        return True

    def _is_edge_free(self, q1, q2, gripper_q=None):
        distance = np.linalg.norm(q2 - q1)
        steps = max(2, int(distance / self.step_size))

        for i in range(steps + 1):
            t = i / steps
            q = q1 + t * (q2 - q1)

            if not self._is_collision_free(q, gripper_q=gripper_q):
                return False

        return True

    def _sample(self, q_goal):
        if np.random.random() < self.goal_bias:
            return q_goal.copy()

        return np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])

    def _nearest(self, tree, q):
        difference = tree.view() - q
        distances = np.sum(difference * difference, axis=1)

        return int(np.argmin(distances))

    def _steer(self, q_near, q_sample):
        difference = q_sample - q_near
        distance = np.linalg.norm(difference)

        if distance <= self.step_size:
            return q_sample.copy()

        return q_near + (difference / distance) * self.step_size

    def _extract_path(self, tree, parents):
        path = []
        index = tree.size - 1

        while index != -1:
            path.append(tree.buffer[index])
            index = parents[index]

        path.reverse()

        return path

    def _shortcut(self, path, gripper_q=None):
        path = list(path)

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

        tree = Tree(np.empty((self.max_iterations + 2, q_start.shape[0]), dtype=float))
        tree.append(q_start)

        parents = [-1]

        for _ in range(self.max_iterations):
            q_sample = self._sample(q_goal)

            closest_index = self._nearest(tree.view(), q_sample)
            q_closest = tree.buffer[closest_index]

            q_new = self.robot_model.clip(self._steer(q_closest, q_sample))

            if not self._is_edge_free(q_closest, q_new, gripper_q=gripper_q):
                continue

            tree.append(q_new)
            parents.append(closest_index)

            if (
                np.linalg.norm(q_new - q_goal) < self.goal_threshold and
                self._is_edge_free(q_new, q_goal, gripper_q=gripper_q)
            ):
                tree.append(q_goal)
                parents.append(tree.size - 2)

                path = self._extract_path(tree, parents)
                path = self._shortcut(path, gripper_q=gripper_q)

                return path

        return None
