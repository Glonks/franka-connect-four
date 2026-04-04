import numpy as np
import mujoco as mj
import xml.etree.ElementTree as ET
import RobotUtil as rt

from mujoco import viewer

from actions import GoTo, GripperState, build_action_sequence, CommonPoses, OpenGripper, CloseGripper
from inverse_kinematics import IKSolver
from scipy.spatial.transform import Rotation as R
import time


ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml" 
MODEL_XML = "franka_emika_panda/panda_torque_table_shelves.xml" 

EndofTable = 0.55 + 0.135 + 0.05

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
    build_env()

    actions = build_action_sequence(None)

    model = mj.MjModel.from_xml_path(MODEL_XML)
    data = mj.MjData(model)

    # Init to Home
    data.qpos[ARM_INDEX] = CommonPoses.Home
    data.qvel[ARM_INDEX] = 0.0
    mj.mj_forward(model, data)

    ik_solver = IKSolver(
        model,
        # max_iterations=1000,
        # step_size=0.1,
        # rotation_tolerance=0.01,
        # W=np.diag([100, 1, 1, 1, 1, 1, 1]),
        # W=np.eye(7),
        # C=np.diag([1e4] * 6)
    )

    # target_position = np.array([EndofTable-0.16, 0.3, 0.315])
    # target_orientation = data.xquat[ik_solver.body_id].copy()
    # target_pose = (target_position, target_orientation)
    # q, success, error = ik_solver.solve(data, target_pose)
    # print(f'{success=}, {error=}')

    v = viewer.launch_passive(model, data)
    v.cam.distance = 3.0
    v.cam.azimuth += 90

    # print(q)
    # print(data.xquat[ik_solver.body_id].copy())

    try:
        while v.is_running():
            input()
            for action in actions:
                print(action)

                t, done = 0.0, False

                while not done:
                    torques, done = action.control(model, data, t)

                    data.ctrl[:7] = torques + data.qfrc_bias[:7]

                    mj.mj_step(model, data)
                    v.sync()

                    t += model.opt.timestep

                print(data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hand")])
                time.sleep(0.1)

            v.sync()

    finally:
        if v is not None:
            v.close()


if __name__ == '__main__':
    main()
