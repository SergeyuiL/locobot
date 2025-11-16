"""implements Whole-Body Control logic with MPC"""

import casadi as ca
from urdf2casadi import urdfparser as u2c
import numpy as np
from scipy.spatial.transform import Rotation
import time
import os.path as osp
import matplotlib.pyplot as plt
import logging, sys
import argparse


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("[%(levelname)5s] [%(funcName)s] %(message)s"))
logger.addHandler(h)
logger.propagate = False


q_ref = [0, -1.1, 1.55, 0, 0.5, 0]  # arm_sleep state, used as a reference for posture regulation


def get_traj_ref(t0, dT, N):
    """
    Generate the reference trajectory and command for the next N steps.

    Parameters
    ----------
    t0 : float
        Current time (not used in the current simple implementation but
        kept for extensibility).
    dT : float
        Time step.
    N : int
        Number of future steps to generate.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 16) where each row is the flattened 4x4
        transformation matrix for the reference pose at each horizon
        step (row-major / C-order)
    """

    T0 = np.eye(4)
    T0[:3, :3] = Rotation.from_euler("xyz", [0, 0, np.pi / 3]).as_matrix()  # 90 degrees rotation around z-axis
    T0[:3, 3] = np.array([2, 0, 0.1])  # target position at origin
    traj = T0[None, :, :].repeat(N, axis=0)

    # t = np.linspace(t0, t0 + 0.1 * dT * N, N)
    # for i in range(N):
    #     t_i = t[i]
    #     traj[i, :3, 3] = np.array([3 * np.sin(t_i), 3 * np.cos(t_i), 0.5])

    return traj.reshape(N, -1)


def state_func(x, u, k):
    """
    Estimated discrete-time state update for the robot.

    Parameters
    ----------
    x : numpy.ndarray or array-like, shape (n,)
        Current state vector [x, y, theta, q1..q6].
    u : numpy.ndarray or array-like, shape (p,)
        Control input vector [v, omega, dq1..dq6].
    k : int
        Current time step index (not used in the simple model but
        provided for compatibility with time-varying models).

    Returns
    -------
    numpy.ndarray
        Next state vector (same shape as `x`).
    """

    global dT
    x1 = np.copy(x)
    x1[0] = x[0] + u[0] * np.cos(x[2]) * dT
    x1[1] = x[1] + u[0] * np.sin(x[2]) * dT
    x1[2] = x[2] + u[1] * dT
    x1[3:] = x[3:] + u[2:] * dT
    return x1


def get_pose_error(T, T_ref):
    """Compute pose error between T and T_ref"""
    err_pos = T[:3, 3] - T_ref[:3, 3]
    R_err = T[:3, :3].T @ T_ref[:3, :3]
    err_rot = R_err - np.eye(3)
    err_rot_vec = np.array([err_rot[2, 1], err_rot[0, 2], err_rot[1, 0]])
    return np.concatenate([err_pos, err_rot_vec], axis=0)


# === MPC 类 ===
class LocobotMPC:
    def __init__(self, dT, mpc_horizon, v_max=0.5, w_max=np.pi / 6, dq_max=np.pi / 3, Kp=10, Ki=1):
        """
        Initialize Locobot MPC with integral action.

        Parameters
        ----------
        k_int : float
            Weight for integral term in cost function.
        """
        self.dT = dT
        self.N = mpc_horizon

        # Original state: [x, y, theta, q1..q6] -> 9D
        self.n_orig = 9
        # Augmented state: [orig_state, z_integral] -> 15D (6D pose error integral)
        self.n_aug = self.n_orig + 6
        self.p = 2 + 6  # [v, omega, dq1..dq6]

        # Load URDF
        path_locobot_urdf = r"../urdf/locobot.urdf"
        parser = u2c.URDFparser()
        parser.from_file(path_locobot_urdf)

        # Build forward kinematics expressions
        root = "locobot/base_link"
        tip = "locobot/gripper_link"
        joint_list, joint_names, q_max, q_min = parser.get_joint_info(root, tip)
        n_joints = parser.get_n_joints(root, tip)
        assert n_joints == 6, "Expected 6 joints for locobot arm, got {}".format(n_joints)
        logger.info(f"joint information: {joint_names}\nq_max: {q_max}\nq_min: {q_min}")
        self.fk_dict = parser.get_forward_kinematics(root, tip)

        # define CasADi optimization problem
        self.opti = ca.Opti()

        # input conditions of self.opti
        self.x0_aug = self.opti.parameter(self.n_aug)  # Augmented initial state
        self.T_ref = self.opti.parameter(
            self.N, 16
        )  # Reference trajectory. 16 is the flatten of 4x4 row-major matrix (e.g., numpy)
        self.k = self.opti.parameter(1)

        # variables to solve
        self.X_aug = self.opti.variable(self.N + 1, self.n_aug)
        self.U = self.opti.variable(self.N, self.p)

        # define solution constraint, goal and solving options.
        self.set_constraints(v_max, w_max, dq_max)
        self.set_cost(Kp, Ki)
        self.set_options()

    def pose2d_to_mat(self, x, y, theta):
        """CasADi version of pose2d_to_mat"""
        R = ca.vertcat(
            ca.horzcat(ca.cos(theta), -ca.sin(theta), 0),
            ca.horzcat(ca.sin(theta), ca.cos(theta), 0),
            ca.horzcat(0, 0, 1),
        )
        t = ca.vertcat(x, y, 0)
        T = ca.vertcat(ca.horzcat(R, t), ca.horzcat(0, 0, 0, 1))
        return T

    def get_gripper_pose(self, X):
        """Get EE pose from original state (first 9 elements)"""
        T_b2w = self.pose2d_to_mat(X[0], X[1], X[2])
        T_ee2b = self.fk_dict["T_fk"](X[3:9])
        return ca.mtimes(T_b2w, T_ee2b)

    def get_pose_error(self, T, T_ref):
        """CasADi version of pose error"""
        err_pos = T[:3, 3] - T_ref[:3, 3]
        R_err = ca.mtimes(T[:3, :3].T, T_ref[:3, :3])
        err_rot = R_err - ca.MX.eye(3)
        err_rot_vec = ca.vertcat(err_rot[2, 1], err_rot[0, 2], err_rot[1, 0])
        return ca.vertcat(err_pos, err_rot_vec)

    def set_constraints(self, v_max, w_max, dq_max):
        # Initial state constraint (full augmented state)
        self.opti.subject_to(self.X_aug[0, :] == self.x0_aug.T)

        # Dynamics and integral update
        for i in range(self.N):
            # Extract current state
            xi = self.X_aug[i, :9].T  # Original state
            zi = self.X_aug[i, 9:15].T  # Integral state
            ui = self.U[i, :].T

            # State dynamics (base + arm)
            x_nxt = ca.vertcat(
                xi[0] + ui[0] * ca.cos(xi[2]) * self.dT,
                xi[1] + ui[0] * ca.sin(xi[2]) * self.dT,
                xi[2] + ui[1] * self.dT,
                xi[3:9] + ui[2:] * self.dT,
            )

            # Pose error for integral update
            Ti = self.get_gripper_pose(xi)
            Ti_ref = ca.reshape(
                self.T_ref[i, :], 4, 4
            ).T  # reshape the row-major flatten to 4x4 matrix; transpose is needed
            err_pose = self.get_pose_error(Ti, Ti_ref)

            # Integral dynamics: z_{k+1} = z_k + error * dt
            z_nxt = zi + err_pose * self.dT

            # Assemble next augmented state
            x_aug_nxt = ca.vertcat(x_nxt, z_nxt)
            self.opti.subject_to(self.X_aug[i + 1, :].T == x_aug_nxt)

        # Input constraints
        self.opti.subject_to(self.opti.bounded(-v_max, self.U[:, 0], v_max))
        self.opti.subject_to(self.opti.bounded(-w_max, self.U[:, 1], w_max))
        self.opti.subject_to(self.opti.bounded(-dq_max, self.U[:, 2:], dq_max))

    def set_cost(self, Kp, Ki):
        self.cost = 0
        for i in range(self.N):
            Xi_orig = self.X_aug[i + 1, :9]
            Zi = self.X_aug[i + 1, 9:15]
            Ui = self.U[i, :]

            # Pose tracking cost
            Ti_ee2w = self.get_gripper_pose(Xi_orig)
            Ti_ref = ca.reshape(self.T_ref[i, :], 4, 4).T
            err_pos = Ti_ee2w[:3, 3] - Ti_ref[:3, 3]
            err_rot = ca.mtimes(Ti_ee2w[:3, :3].T, Ti_ref[:3, :3]) - ca.MX.eye(3)
            self.cost += (ca.sumsqr(err_pos) + ca.sumsqr(err_rot)) * Kp

            # Posture regulation
            self.cost += ca.sumsqr(Xi_orig[3:].T - q_ref)

            # # Control effort
            # self.cost += ca.sumsqr(Ui)

            # Integral term
            self.cost += ca.sumsqr(Zi) * Ki

        self.opti.minimize(self.cost)

    def set_options(self):
        opts_setting = {
            "jit": False,
            "print_time": 0,
            "ipopt.print_level": 0,  # Set to 0 for less verbose output
        }
        self.opti.solver("ipopt", opts_setting)

    def solve(self, x0_aug, T_ref, k, warm_start=True):
        """
        Solve MPC with augmented initial state.

        Parameters
        ----------
        x0_aug : array-like, shape (15,)
            [x, y, theta, q1..q6, zx, zy, zz, zrx, zry, zrz]

        Returns
        -------
        X_aug_sol : np.ndarray, shape (N+1, 15)
            Full augmented state trajectory.
        U_sol : np.ndarray, shape (N, 8)
            Control trajectory.
        """
        if warm_start and hasattr(self, "sol"):
            self.opti.set_initial(self.X_aug, self.sol.value(self.X_aug))
            self.opti.set_initial(self.U, self.sol.value(self.U))

        self.opti.set_value(self.x0_aug, x0_aug)
        self.opti.set_value(self.T_ref, T_ref)
        self.opti.set_value(self.k, k)

        self.sol = self.opti.solve()

        X_aug = self.sol.value(self.X_aug)
        U = self.sol.value(self.U)
        n1, n2, p = self.n_orig, self.n_aug, self.p
        N = self.N
        return X_aug.reshape(N + 1, n2)[:, :n1], U.reshape(N, p)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sim", action="store_true", help="Use simulation mode with LocobotSocket")
    args = argparser.parse_args()
    logging.info(f"Simulation mode: {args.sim}")

    dT = 0.1
    N = 10
    locobot_ctl = LocobotMPC(dT=dT, mpc_horizon=N, Kp=40, Ki=40)
    k = 0
    x = np.array([0, 0, 0, *q_ref])  # [x, y, theta, q1, q2, q3, q4, q5, q6]
    z = np.zeros(locobot_ctl.n_aug - locobot_ctl.n_orig)  # [z1, z2, z3, z4, z5, z6]
    sim_time = 10
    x_list, u_list, t_list, T_list = [], [], [], []
    Te_list = []
    cost_list = []
    solv_time_list = []

    if args.sim:
        import LocobotSocket

        sock = LocobotSocket.LocobotSocket("172.27.80.1", 12345)
        sock.send_command(reset=1)  # Reset the robot in Unity
    while 1:
        t = k * dT
        if t > sim_time:
            break
        T_ref = get_traj_ref(t, dT, N)
        # get the control with mpc controller
        t1 = time.time()
        x_aug = np.concatenate([x, z], axis=0)
        X, U = locobot_ctl.solve(x_aug, T_ref, k, warm_start=True)
        solv_time_list.append(time.time() - t1)

        Ti = np.asarray(locobot_ctl.get_gripper_pose(x))
        Ti_ref = T_ref[0].reshape(4, 4)
        Ti_err = np.linalg.inv(Ti) @ Ti_ref

        # record input, state and time
        u_list.append(U[0, :])
        t_list.append(t)
        x_list.append(x)
        cost_list.append(locobot_ctl.opti.value(locobot_ctl.cost))
        T_list.append(Ti)
        Te_list.append(Ti_err)

        # simulate state update
        x = state_func(x[:9], U[0, :], k)
        err_vec = get_pose_error(Ti, Ti_ref)
        z = z + err_vec * dT if np.linalg.norm(err_vec) < 0.5 else np.zeros_like(z)  # reset integral if error too large
        k += 1
        if args.sim:
            sock.send_command(
                joint_angles=list(np.rad2deg(X[N - 1, 3:])),  # Send the first joint angles
                velocity=U[0, 0],  # Send the first linear velocity
                angular_velocity=np.rad2deg(U[0, 1]),  # Convert to degrees/s
            )
            time.sleep(max(0, dT - (time.time() - t1)))

    if args.sim:
        sock.send_command(velocity=0, angular_velocity=0)  # Stop the robot
    Te_list = np.array(Te_list)
    pos = Te_list[:, :3, 3]
    rot = [Rotation.from_matrix(Te_list[i, :3, :3]).as_euler("xyz") for i in range(Te_list.shape[0])]
    plt.figure()
    plt.plot(t_list, pos)
    plt.legend(["x", "y", "z"])
    plt.xlabel("time (s)")
    plt.ylabel("error (m)")
    plt.figure()
    plt.plot(t_list, rot)
    plt.legend(["roll", "pitch", "yaw"])
    plt.xlabel("time (s)")
    plt.ylabel("error (rad)")
    plt.show()
