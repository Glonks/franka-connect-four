import numpy as np


def is_cartesian_pose(target):
    # We need to be able to unpack target into position + orientation
    if not isinstance(target, (tuple, list)) or len(target) != 2:
        return False

    position, orientation = target

    # Need 3D position and quaternion
    return (
        np.asarray(position).shape == (3,) and
        np.asarray(orientation).shape == (4,)
    )


def interpolate_min_jerk(q_start, q_goal, t, T):
    tau = np.clip(t / max(T, 1e-6), 0.0, 1.0)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / max(T, 1e-6)

    q_des = q_start + s * (q_goal - q_start)
    qd_des = ds * (q_goal - q_start)
    
    return q_des, qd_des
