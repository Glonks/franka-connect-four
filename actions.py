import numpy as np

from enum import Enum
from dataclasses import dataclass

import RobotUtil as rt


KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float)

HOLD_TIME = 0.5
SEGMENT_TIME = 2.5
GRIPPER_TIME = 0.8


class GripperState(float, Enum):
    Open = 0.04
    Closed = 0.0125


def compute_torques(q_des, qd_des, q, qd):
    return KP * (q_des - q) + KD * (qd_des - qd)


class GoTo:
    def __init__(self, target):
        self.target = np.array(target)
        self._q_start = None
    
    def _plan(self, q):
        self._q_start = q.copy()

    def control(self, model, data, time):
        q = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()

        if self._q_start is None:
            self._plan(q)

        q_des, qd_des = rt.interp_min_jerk(
            self._q_start,
            self.target,
            time,
            SEGMENT_TIME
        )

        torques = compute_torques(q_des, qd_des, q, qd)

        done = time >= (SEGMENT_TIME + HOLD_TIME)

        return torques, done

    def __str__(self) -> str:
        with np.printoptions(precision=4):
            return f'<{self.__class__.__name__} {self.target=}>'


class CloseGripper:
    def __init__(self):
        pass

    def control(self, model, data, time):
        q = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()

        q_des, qd_des = q.copy(), np.zeros_like(qd)

        torques = compute_torques(q_des, qd_des, q, qd)
        data.ctrl[7] = GripperState.Closed

        done = time >= GRIPPER_TIME

        return torques, done
    
    def __str__(self) -> str:
        return f'<{self.__class__.__name__}>'


class OpenGripper:
    def __init__(self):
        pass

    def control(self, model, data, time):
        q = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()

        q_des, qd_des = q.copy(), np.zeros_like(qd)

        torques = compute_torques(q_des, qd_des, q, qd)
        data.ctrl[7] = GripperState.Open

        done = time >= GRIPPER_TIME

        return torques, done

    def __str__(self) -> str:
        return f'<{self.__class__.__name__}>'


@dataclass(frozen=True)
class CommonPoses:
    Home              = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5,  0.8]
    PreInitialGrasp   = [0.0, 0.65, 0.0, -2.0, 0.0, 2.65, 0.8]  # TODO: Construct from initial block pose
    LeftShelfDrop     = []
    PreLeftShelfGrasp = []


def build_action_sequence(intial_block_pose):
    return [
        GoTo(CommonPoses.Home),
        OpenGripper(),
        GoTo(CommonPoses.PreInitialGrasp),
        CloseGripper(),
    ]
