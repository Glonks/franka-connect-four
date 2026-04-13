import numpy as np


class RRTPlanner:
    def __init__(
        self,
        robot_model,
        obstacles,
        step_size=0.15,
        max_iterations=5000,
        goal_bias=0.2,
        goal_threshold=0.3,
    ):
        self.robot_model = robot_model
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.goal_threshold = goal_threshold
        self.joint_limits = self.robot_model.joint_limits.copy()

        self.obs_centers = np.array([o[1] for o in obstacles], dtype=float)
        self.obs_halfs = np.array([o[2] for o in obstacles], dtype=float)

    def _is_collision_free(self, q):
        return not self.robot_model.collides_with_boxes(q, self.obs_centers, self.obs_halfs)

    def _is_edge_free(self, q_1, q_2, num_checks=10):
        for t in np.linspace(0.0, 1.0, num_checks):
            q = q_1 + t * (q_2 - q_1)

            if not self._is_collision_free(q):
                return False

        return True

    def _sample(self, q_goal):
        if np.random.random() < self.goal_bias:
            return q_goal.copy()

        return np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])

    def _nearest(self, tree, q):
        distances = np.linalg.norm(tree - q, axis=1)

        return int(np.argmin(distances))

    def _steer(self, q_near, q_sample):
        difference = q_sample - q_near
        distance = np.linalg.norm(difference)

        if distance <= self.step_size:
            return q_sample.copy()

        return q_near + (difference / distance) * self.step_size

    def _extract_path(self, tree, parents):
        path = []
        index = len(tree) - 1

        while index != -1:
            path.append(tree[index])
            index = parents[index]

        path.reverse()

        return path

    def _shortcut(self, path, attempts=200):
        path = list(path)
        for _ in range(attempts):
            if len(path) <= 2:
                break

            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))

            if self._is_edge_free(path[i], path[j], num_checks=100):
                path = path[: i + 1] + path[j:]

        return path

    def plan(self, q_start, q_goal):
        q_start = self.robot_model.clip(np.asarray(q_start, dtype=float))
        q_goal = self.robot_model.clip(np.asarray(q_goal, dtype=float))

        if not self._is_collision_free(q_start):
            print("Start configuration is in collision")
            return None

        if not self._is_collision_free(q_goal):
            print("Goal configuration is in collision")
            return None

        if self._is_edge_free(q_start, q_goal, num_checks=100):
            return [q_start.copy(), q_goal.copy()]

        tree = [q_start.copy()]
        parents = [-1]

        for _ in range(self.max_iterations):
            q_sample = self._sample(q_goal)
            closest_index = self._nearest(np.array(tree), q_sample)
            q_closest = tree[closest_index]
            q_new = self.robot_model.clip(self._steer(q_closest, q_sample))

            if not self._is_edge_free(q_closest, q_new):
                continue

            tree.append(q_new)
            parents.append(closest_index)

            if (
                np.linalg.norm(q_new - q_goal) < self.goal_threshold and
                self._is_edge_free(q_new, q_goal)
            ):
                tree.append(q_goal.copy())
                parents.append(len(tree) - 2)

                path = self._extract_path(tree, parents)
                path = self._shortcut(path)

                return path

        return None
