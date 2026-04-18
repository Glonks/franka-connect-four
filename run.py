import numpy as np
import xml.etree.ElementTree as ET
import RobotUtil as rt

from actions import build_action_sequence, CommonPoses
from kinematics import PandaKinematics
from inverse_kinematics import IKSolver
from motion_planning import RRTPlanner
from runtime import MujocoRuntime
import time
import mujoco as mj
from mujoco import viewer


ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"

# True: load pre-built Lab 3 scene (run `python lab3/build_lab3_xml.py` first).
# False: generate connect-four + shelves into MODEL_XML via build_env().
USE_LAB3_SCENE = True
MODEL_XML = (
    "franka_emika_panda/panda_torque_table_lab3.xml"
    if USE_LAB3_SCENE
    else "franka_emika_panda/panda_torque_table_shelves.xml"
)

EndofTable = 0.55 + 0.135 + 0.05

# Obstacles for RRT when using the Lab 3 MJCF (table only; no shelf boxes).
BLOCKS_LAB3 = [
    ["TablePlane", [EndofTable - 0.275, 0.0, -0.005], [0.275, 0.504, 0.0051]],
]

BLOCKS=[
    ["TablePlane",[EndofTable-0.275,0.,-0.005],[0.275, 0.504, 0.0051]],
    ["LShelfDistal",[EndofTable-0.09-0.0225, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
    ["LShelfProximal",[EndofTable-0.55-0.0225, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
    ["LShelfBack",[EndofTable-0.55-0.0225-0.09, 0.504-0.045-0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
    ["LShelfMid",[EndofTable-0.32, 0.504-0.045-0.0225, 0.315],[0.0225, 0.0225, 0.315]],
    ["LShelfArch",[EndofTable-0.275-0.135+0.0225, 0.504-0.045-0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
    ["LShelfBottom",[EndofTable-0.275-0.135+0.0225, 0.504-0.09-0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
    ["LShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSupp2",[EndofTable-0.32-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, 0.504-0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, 0.504-0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
    ["RShelfDistal",[EndofTable-0.09-0.0225, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
    ["RShelfProximal",[EndofTable-0.55-0.0225, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
    ["RShelfBack",[EndofTable-0.55-0.0225-0.09, -0.504+0.045+0.0225, 0.3825-0.135],[0.0225, 0.0225, 0.3825]],
    ["RShelfMid",[EndofTable-0.32, -0.504+0.045+0.0225, 0.315],[0.0225, 0.0225, 0.315]],
    ["RShelfArch",[EndofTable-0.275-0.135+0.0225, -0.504+0.045+0.0225, 0.63+0.0225],[0.315, 0.0225, 0.0225]],
    ["RShelfBottom",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005],[0.2525, 0.135/2., 0.005]],
    ["RShelfBottomSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225],[0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225],[0.315, 0.0225, 0.0225]],
    ["RShelfMiddle",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.2],[0.2525, 0.135/2., 0.005]],
    ["RShelfMiddleSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],[0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.2],[0.315, 0.0225, 0.0225]],
    ["RShelfTop",[EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.4],[0.2525, 0.135/2., 0.005]],
    ["RShelfTopSupp1",[EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
    ["RShelfTopSupp2",[EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
    ["RShelfTopSupp3",[EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],[0.0225, 0.1125, 0.0225]],
    ["RShelfTopSuppB",[EndofTable-0.275-0.135+0.0225, -0.504+0.0225,0.1375+0.0225+.4],[0.315, 0.0225, 0.0225]],
]

ARM_INDEX = [0, 1, 2, 3, 4, 5, 6]


def build_env():
    global BLOCKS, EndofTable

    modelTree = ET.parse(ROOT_MODEL_XML)

    for block in BLOCKS:
        rt.add_free_block_to_model(
            tree=modelTree,
            name=block[0],
            pos=block[1],
            density=20,
            size=block[2],
            rgba=[0.2, 0.2, 0.9, 1],
            free=False
        )
    
    rt.add_free_block_to_model(
        tree=modelTree,
        name="Block",
        pos=[
            EndofTable - 0.145,
            0.0,
            0.05
        ],
        density=20,
        size=[0.02, 0.02, 0.02],
        rgba=[0.0, 0.9, 0.2, 1],
        free=True
    )

    modelTree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)


def main():
    if not USE_LAB3_SCENE:
        build_env()

    robot_model = PandaKinematics(ROOT_MODEL_XML)
    ik_solver = IKSolver(robot_model)
    planner_blocks = BLOCKS_LAB3 if USE_LAB3_SCENE else BLOCKS
    planner = RRTPlanner(robot_model, planner_blocks, step_size=0.05)

    # Demo trajectory is still the connect-four / shelf sequence; replace when you add Lab 3 actions.
    actions = build_action_sequence(robot_model, ik_solver, planner)

    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)
    runtime = MujocoRuntime(model, data, ARM_INDEX)
    runtime.set_configuration(CommonPoses.Home)

    with viewer.launch_passive(model, data) as v:
        v.cam.distance = 3.0
        v.cam.azimuth += 90

        for action in actions:
            print(action)

            t, done = 0.0, False

            while not done:
                if not v.is_running():
                    break

                command, done = action.control(runtime.get_state(), t)
                runtime.step(command)
                v.sync()

                t += runtime.dt

            time.sleep(0.1)
    
    time.sleep(0.1)


if __name__ == '__main__':
    main()
