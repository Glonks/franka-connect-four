import mujoco as mj
import numpy as np

from dataclasses import dataclass


KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float) * 1.1
KD = np.array([8, 8, 6, 5, 4, 3, 2], dtype=float) * 10


@dataclass(frozen=True)
class RobotState:
    q: np.ndarray
    qd: np.ndarray
    gripper_q: np.ndarray


@dataclass(frozen=True)
class ControlCommand:
    q_des: np.ndarray
    qd_des: np.ndarray
    gripper_target: float | None = None


class MujocoRuntime:
    def __init__(self, model, data, arm_indices, gripper_index=7):
        self.model = model
        self.data = data
        self.arm_indices = list(arm_indices)
        self.gripper_index = gripper_index

    @property
    def dt(self):
        return self.model.opt.timestep

    def set_configuration(self, q):
        self.data.qpos[self.arm_indices] = q
        self.data.qvel[self.arm_indices] = 0.0

        mj.mj_forward(self.model, self.data)

    def get_state(self):
        return RobotState(
            q=self.data.qpos[self.arm_indices].copy(),
            qd=self.data.qvel[self.arm_indices].copy(),
            gripper_q=self.data.qpos[7:9].copy(),
        )

    def step(self, command: ControlCommand):
        state = self.get_state()
        torques = KP * (command.q_des - state.q) + KD * (command.qd_des - state.qd)

        self.data.ctrl[:7] = torques + self.data.qfrc_bias[:7]

        if command.gripper_target is not None:
            self.data.ctrl[self.gripper_index] = command.gripper_target

        mj.mj_step(self.model, self.data)


class FrankaPyRuntime:
    def __init__(self):
        raise NotImplementedError
