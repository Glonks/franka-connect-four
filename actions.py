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
    PreInitialGrasp   = [0.0, 0.64928, 0.0, -2.0328, 0.0, 2.6822, 0.8]  # TODO: Construct from initial block pose
    ResetPoint        = [0.0, -0.69227,  0.0, -2.6583,  0.0, 1.9659,  0.8]
    
    LeftShelfAbove1 = [ 0.2329, -0.2621,  0.5974, -2.6448,  0.2213,  2.4143,  1.4477]
    LeftShelfPreGrasp1 = [ 0.1515,  0.1515,  0.0049, -2.8248,  1.7249,  1.5972,  0.9632]
    LeftShelfGrasp1 = [ 0.2346,  0.2121,  0.0905, -2.7754,  1.8876,  1.6378,  0.9413]
    RightShelf1 = [-0.1888,  0.3682, -0.0474, -2.4325, -1.7854,  1.6639,  0.4707]
    
    LeftShelfAbove2 = [0.1994, 0.058, 0.4267, -2.3298, -0.0346, 2.3824, 1.4504]
    LeftShelfPreGrasp2 = [0.1234, 0.353, -0.0128, -2.4606, 1.6778, 1.6026, 1.1266]
    LeftShelfGrasp2 = [0.1873, 0.3976, 0.0479, -2.4216, 1.7852, 1.6609, 1.1109]
    RightShelf2 = [-0.1662, -0.2031, -0.0871, -2.4902, -1.7509, 1.7481, -0.04]
    
    LeftShelfAbove3 = [ 0.2447,  0.4316,  0.2632, -1.7997, -0.1364,  2.2139,  1.3668]
    LeftShelfPreGrasp3 = [ 0.1045,  0.6425, -0.0262, -1.9396,  1.65  ,  1.6019,  1.359 ]
    LeftShelfGrasp3 = [ 0.1613,  0.6787,  0.0144, -1.9042,  1.7129,  1.6696,  1.3509]
    RightShelf3 = [-0.0398, -0.4243, -0.1793, -2.1716, -1.6804,  1.7576, -0.5856]


def build_action_sequence(poses):
    return [
        GoTo(CommonPoses.Home),
        OpenGripper(),
        GoTo(CommonPoses.PreInitialGrasp),
        CloseGripper(),
        GoTo(CommonPoses.ResetPoint),
        GoTo(CommonPoses.LeftShelfAbove1),
        # GoTo(CommonPoses.LeftShelfAbove2),
        # GoTo(CommonPoses.LeftShelfAbove3),
        OpenGripper(),
        GoTo(CommonPoses.ResetPoint),
        GoTo(CommonPoses.LeftShelfPreGrasp1),
        GoTo(CommonPoses.LeftShelfGrasp1),
        # GoTo(CommonPoses.LeftShelfPreGrasp2),
        # GoTo(CommonPoses.LeftShelfGrasp2),
        # GoTo(CommonPoses.LeftShelfPreGrasp3),
        # GoTo(CommonPoses.LeftShelfGrasp3),
        CloseGripper(),
        GoTo(CommonPoses.ResetPoint),
        GoTo(CommonPoses.RightShelf1),
        # GoTo(CommonPoses.RightShelf2),
        # GoTo(CommonPoses.RightShelf3),
        OpenGripper(),
        GoTo(CommonPoses.Home)
    ]
