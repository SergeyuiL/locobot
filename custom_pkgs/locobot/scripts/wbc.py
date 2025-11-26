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

np.set_printoptions(precision=3, suppress=True)


class WBC:
    def __init__(self, root="locobot/base_footprint", tip="locobot/ee_gripper_link", weights=None):
        # Load URDF
        path_locobot_urdf = r"../urdf/locobot.urdf"
        parser = u2c.URDFparser()
        parser.from_file(path_locobot_urdf)

        # Build forward kinematics expressions
        joint_list, joints_name, q_max, q_min = parser.get_joint_info(root, tip)
        q_max = np.array(q_max)
        q_min = np.array(q_min)
        n_joints = parser.get_n_joints(root, tip)
        assert n_joints == 6, "Expected 6 joints for locobot arm, got {}".format(n_joints)
        print(f"joints name: {joints_name}\nq_max: {q_max}\nq_min: {q_min}")
        self.fk_dict = parser.get_forward_kinematics(root, tip)

        n = 3 + 6  # (x, y, theta, q1, q2, ..., q6)

        self.weights = np.ones(n) if weights is None else np.array(weights)

        # define CasADi optimization problem
        self.opti = ca.Opti()
        # goal pose
        self.T_ref = self.opti.parameter(4, 4)
        self.x0 = self.opti.parameter(n)
        # variables to solve
        self.x = self.opti.variable(n)

        # constrain gripper pose
        T = self.get_gripper_pose(self.x)
        self.opti.subject_to(ca.sumsqr(T - self.T_ref) <= 1e-4)  # position and orientation error within threshold
        # q_min[1] = -1.1
        self.opti.subject_to(self.opti.bounded(q_min, self.x[3:], q_max))  # joint limits

        # set joints cost
        dx = self.x - self.x0
        self.cost = ca.dot(self.weights * dx, dx)
        self.opti.minimize(self.cost)

        # set solver options
        opts_setting = {
            "jit": True,
            "print_time": 0,
            "ipopt.print_level": 0,  # Set to 0 for less verbose output
        }
        self.opti.solver("ipopt", opts_setting)

    def pose2d_to_mat(self, x, y, theta):
        """CasADi version of pose2d_to_mat"""
        R = ca.vertcat(
            ca.horzcat(ca.cos(theta), -ca.sin(theta), 0),
            ca.horzcat(ca.sin(theta), ca.cos(theta), 0),
            ca.horzcat(0, 0, 1),
        )
        t = ca.vertcat(x, y, -0.242)
        T = ca.vertcat(ca.horzcat(R, t), ca.horzcat(0, 0, 0, 1))
        return T

    def get_gripper_pose(self, X):
        """Get EE pose from original state (first 9 elements)"""
        T_b2w = self.pose2d_to_mat(X[0], X[1], X[2])
        T_ee2b = self.fk_dict["T_fk"](X[3:9])
        return ca.mtimes(T_b2w, T_ee2b)

    def solve(self, x0, T_ref):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.T_ref, T_ref)
        self.sol = self.opti.solve()
        x = self.sol.value(self.x)
        return x


class CartMPC:
    def __init__(self, mpc_horizon=20, dt=0.05, vmax=0.3, wmax=0.5, dvmax=0.2, dwmax=0.2):
        self.dt = dt
        self.N = mpc_horizon

        self.opti = ca.Opti()
        # define parameters
        self.x_ref = self.opti.parameter(3)
        self.x0 = self.opti.parameter(3)
        self.z0 = self.opti.parameter(3)
        # define variables
        self.U = self.opti.variable(self.N, 2)
        self.X = self.opti.variable(self.N + 1, 3)
        self.Z = self.opti.variable(self.N + 1, 3)

        # set constraint
        self.opti.subject_to(self.X[0, :].T == self.x0)
        self.opti.subject_to(self.Z[0, :].T == self.z0)
        for k in range(self.N):
            x_k1 = self.X[k + 1, :].T
            x_k = self.X[k, :].T
            u_k = self.U[k, :]
            self.opti.subject_to(
                x_k1 == x_k + self.dt * ca.vertcat(u_k[0] * ca.cos(x_k[2]), u_k[0] * ca.sin(x_k[2]), u_k[1])
            )

            z_k1 = self.Z[k + 1, :].T
            z_k = self.Z[k, :].T
            dx = x_k - self.x_ref
            self.opti.subject_to(z_k1 == z_k + dx * self.dt)

        if self.N > 1:
            self.opti.subject_to(self.opti.bounded(-dvmax * dt, self.U[1:, 0] - self.U[:-1, 0], dvmax * dt))
            self.opti.subject_to(self.opti.bounded(-dwmax * dt, self.U[1:, 1] - self.U[:-1, 1], dwmax * dt))

        self.opti.subject_to(self.opti.bounded(-vmax, self.U[:, 0], vmax))
        self.opti.subject_to(self.opti.bounded(-wmax, self.U[:, 1], wmax))

        # set cost
        self.cost = 0
        for k in range(self.N):
            dx = self.X[k, :].T - self.x_ref
            self.cost += ca.dot(dx, dx)
            self.cost += ca.dot(self.Z[k, :], self.Z[k, :]) * 0.5  # integral action
        self.opti.minimize(self.cost)

        # set solver options
        opts_setting = {
            "jit": True,
            "print_time": 0,
            "ipopt.print_level": 0,  # Set to 0 for less verbose
        }
        self.opti.solver("ipopt", opts_setting)

    def solve(self, x0, z0, x_ref, warm_start=True):
        if warm_start and hasattr(self, "sol"):
            self.opti.set_initial(self.X, self.sol.value(self.X))
            self.opti.set_initial(self.Z, self.sol.value(self.Z))
            self.opti.set_initial(self.U, self.sol.value(self.U))
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.z0, z0)
        self.opti.set_value(self.x_ref, x_ref)
        self.sol = self.opti.solve()
        u0 = self.sol.value(self.U[0, :])
        return u0
