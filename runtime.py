import mujoco as mj
import numpy as np

from dataclasses import dataclass

try:
    from frankapy import FrankaArm
    _FRANKAPY_AVAILABLE = True
except Exception:
    FrankaArm = None
    _FRANKAPY_AVAILABLE = False


def frankapy_available() -> bool:
    return _FRANKAPY_AVAILABLE


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
    KP = np.array([132, 132, 110,  99,  66,  44,  33], dtype=float) 
    KD = np.array([80, 80, 60, 50, 40, 30, 20], dtype=float)

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
        torques = self.KP * (command.q_des - state.q) + self.KD * (command.qd_des - state.qd)

        self.data.ctrl[:7] = torques + self.data.qfrc_bias[:7]

        if command.gripper_target is not None:
            self.data.ctrl[self.gripper_index] = command.gripper_target

        mj.mj_step(self.model, self.data)


class FrankaPyRuntime:
    INIT_DURATION = 2.0
    CONTROL_DT = 0.1

    def __init__(self, model, data, arm_indices, gripper_index=7):
        if not frankapy_available():
            raise RuntimeError('frankapy not available in this environment')

        self.model = model
        self.data = data
        self.arm_indices = list(arm_indices)
        self.gripper_index = gripper_index

        self.robot = FrankaArm()
        self.last_gripper_target = None

    @property
    def dt(self):
        return self.CONTROL_DT

    def set_configuration(self, q):
        self.robot.goto_joints(
            np.asarray(q, dtype=float).tolist(),
            duration=self.INIT_DURATION,
            block=True
        )

    def get_state(self):
        state = self.robot.get_robot_state()

        return RobotState(
            q=np.asarray(state['joints'], dtype=float),
            qd=np.asarray(state['joint_velocities'], dtype=float),
            gripper_q=(float(state['gripper_width']) / 2.0) * np.ones(2, dtype=float)
        )

    def step(self, command: ControlCommand):
        q_des = np.asarray(command.q_des, dtype=float).tolist()
        self.robot.goto_joints(q_des, duration=self.dt, block=True)

        if command.gripper_target is not None:
            target = float(command.gripper_target)
            if (
                self.last_gripper_target is None
                or abs(target - self.last_gripper_target) > 1e-4
            ):
                if target >= 0.03:
                    self.robot.open_gripper(block=True)
                else:
                    self.robot.goto_gripper(target, block=True)

                self.last_gripper_target = target
