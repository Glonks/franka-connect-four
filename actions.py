import functools
import numpy as np

from dataclasses import dataclass
from enum import Enum
from runtime import ControlCommand
from utils import is_cartesian_pose, interpolate_min_jerk


DEFAULT_SEGMENT_SPEED = 0.75
MIN_SEGMENT_TIME = 0.6
HOLD_TIME = 0.5
GRIPPER_TIME = 0.8


class GripperState(float, Enum):
    Open = 0.04
    Closed = 0.0125


@dataclass(frozen=True)
class CommonPoses:
    Home = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]
    PreInitialGrasp = [0.0, 0.64928, 0.0, -2.0328, 0.0, 2.6822, 0.8]
    ResetPoint = [0.0, -0.69227, 0.0, -2.6583, 0.0, 1.9659, 0.8]

    LeftShelfAbove1 = [0.2329, -0.2621, 0.5974, -2.6448, 0.2213, 2.4143, 1.4477]
    LeftShelfPreGrasp1 = [0.1515, 0.1515, 0.0049, -2.8248, 1.7249, 1.5972, 0.9632]
    LeftShelfGrasp1 = [0.2346, 0.2121, 0.0905, -2.7754, 1.8876, 1.6378, 0.9413]
    RightShelf1 = [-0.1888, 0.3682, -0.0474, -2.4325, -1.7854, 1.6639, 0.4707]

    LeftShelfAbove2 = [0.1994, 0.058, 0.4267, -2.3298, -0.0346, 2.3824, 1.4504]
    LeftShelfPreGrasp2 = [0.1234, 0.353, -0.0128, -2.4606, 1.6778, 1.6026, 1.1266]
    LeftShelfGrasp2 = [0.1873, 0.3976, 0.0479, -2.4216, 1.7852, 1.6609, 1.1109]
    RightShelf2 = [-0.1662, -0.2031, -0.0871, -2.4902, -1.7509, 1.7481, -0.04]

    LeftShelfAbove3 = [0.2447, 0.4316, 0.2632, -1.7997, -0.1364, 2.2139, 1.3668]
    LeftShelfPreGrasp3 = [0.1045, 0.6425, -0.0262, -1.9396, 1.65, 1.6019, 1.359]
    LeftShelfGrasp3 = [0.1613, 0.6787, 0.0144, -1.9042, 1.7129, 1.6696, 1.3509]
    RightShelf3 = [-0.0398, -0.4243, -0.1793, -2.1716, -1.6804, 1.7576, -0.5856]


class JointWaypointTrajectory:
    def __init__(
        self,
        waypoints,
        segment_speed=DEFAULT_SEGMENT_SPEED,
        min_segment_time=MIN_SEGMENT_TIME
    ):
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        if self.waypoints.ndim != 2 or self.waypoints.shape[0] < 2:
            raise ValueError("Trajectory requires at least two joint-space waypoints")

        deltas = np.diff(self.waypoints, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)

        self.segment_times = np.maximum(
            segment_lengths / max(segment_speed, 1e-6),
            min_segment_time
        )
        self.times = np.concatenate([[0.0], np.cumsum(self.segment_times)])
        self.duration = float(self.times[-1])

    def sample(self, time):
        if time <= 0.0:
            return self.waypoints[0].copy(), np.zeros(self.waypoints.shape[1])

        if time >= self.duration:
            return self.waypoints[-1].copy(), np.zeros(self.waypoints.shape[1])

        segment_index = np.searchsorted(self.times, time, side="right") - 1
        segment_index = min(max(segment_index, 0), len(self.segment_times) - 1)

        local_time = time - self.times[segment_index]

        return interpolate_min_jerk(
            self.waypoints[segment_index],
            self.waypoints[segment_index + 1],
            local_time,
            self.segment_times[segment_index],
        )


class GoTo:
    def __init__(
        self,
        target,
        robot_model,
        ik_solver,
        planner,
        bias=None,
        hold_time=HOLD_TIME,
        segment_speed=DEFAULT_SEGMENT_SPEED,
        min_segment_time=MIN_SEGMENT_TIME,
    ):
        self.target = target
        self.robot_model = robot_model
        self.ik_solver = ik_solver
        self.planner = planner
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float64)
        self.hold_time = hold_time
        self.segment_speed = segment_speed
        self.min_segment_time = min_segment_time
        self._trajectory = None

    def _resolve_goal(self, q_start):
        if is_cartesian_pose(self.target):
            q_goal, success, error = self.ik_solver.solve(
                q_start,
                self.target,
                bias=self.bias
            )

            if not success:
                raise RuntimeError(f"IK failed with residual {error}")

            return q_goal

        return self.robot_model.clip(np.asarray(self.target, dtype=np.float64))

    def _build_waypoints(self, q_start, q_goal, gripper_q):
        path = self.planner.plan(q_start, q_goal, gripper_q=gripper_q)
        if path is None:
            raise RuntimeError("Motion planner failed to find a collision-free path")

        return path

    def _plan(self, state):
        q_start = state.q.copy()
        q_goal = self._resolve_goal(q_start)

        waypoints = self._build_waypoints(q_start, q_goal, state.gripper_q.copy())

        self._trajectory = JointWaypointTrajectory(
            waypoints,
            segment_speed=self.segment_speed,
            min_segment_time=self.min_segment_time,
        )

    def control(self, state, time):
        if self._trajectory is None:
            self._plan(state)

        q_des, qd_des = self._trajectory.sample(time)
        done = time >= (self._trajectory.duration + self.hold_time)

        return ControlCommand(q_des=q_des, qd_des=qd_des), done

    def __str__(self):
        with np.printoptions(precision=4):
            if is_cartesian_pose(self.target):
                position, orientation = map(np.asarray, self.target)
                return f'<{self.__class__.__name__} position={position} orientation={orientation}>'

            return f'<{self.__class__.__name__} configuration={np.asarray(self.target)}>'


class GripperAction:
    def __init__(
        self,
        gripper_target, 
        duration=GRIPPER_TIME
    ):
        self.gripper_target = gripper_target
        self.duration = duration
        self.hold_q = None
    
    def control(self, state, time):
        if self.hold_q is None:
            self.hold_q = state.q.copy()

        command = ControlCommand(
            q_des=self.hold_q.copy(),
            qd_des=np.zeros_like(state.qd),
            gripper_target=self.gripper_target
        )
        done = time >= self.duration

        return command, done

    def __str__(self):
        with np.printoptions(precision=4):
            return f'<{self.__class__.__name__} target={self.gripper_target}>'


def build_action_sequence(robot_model, ik_solver, planner):
    def _pose_for(q):
        pose = robot_model.forward_kinematics(q)
        return pose.position, pose.orientation

    _GoTo = functools.partial(
        GoTo,
        robot_model=robot_model,
        ik_solver=ik_solver,
        planner=planner
    )

    _OpenGripper = functools.partial(
        GripperAction,
        gripper_target=GripperState.Open
    )

    _CloseGripper = functools.partial(
        GripperAction,
        gripper_target=GripperState.Closed
    )

    return [
        _GoTo(CommonPoses.Home),
        _OpenGripper(),
        _GoTo(_pose_for(CommonPoses.PreInitialGrasp)),
        _CloseGripper(),
        _GoTo(_pose_for(CommonPoses.ResetPoint)),
        _GoTo(_pose_for(CommonPoses.LeftShelfAbove1)),
        _OpenGripper(),
        _GoTo(_pose_for(CommonPoses.ResetPoint)),
        _GoTo(_pose_for(CommonPoses.LeftShelfPreGrasp1)),
        _GoTo(_pose_for(CommonPoses.LeftShelfGrasp1)),
        _CloseGripper(),
        _GoTo(_pose_for(CommonPoses.ResetPoint)),
        _GoTo(_pose_for(CommonPoses.RightShelf1)),
        _OpenGripper(),
        _GoTo(CommonPoses.Home),
    ]
