#!/usr/bin/env python3

from typing import Callable, List, Optional, Tuple

from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, TransformStamped
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import RobotTrajectory, RobotState
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState

import numpy as np
import rospy
from tf2_ros import Buffer, TransformListener

import moveit_commander
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
import sys

from geometry_msgs.msg import PoseStamped, Pose
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from locobot.srv import SetPose, SetPoseRequest, SetPoseResponse
from locobot.srv import SetFloat32, SetFloat32Request, SetFloat32Response
from locobot.srv import SetPoseArray, SetPoseArrayRequest, SetPoseArrayResponse


np.set_printoptions(precision=3, suppress=True)


def _tuple_to_pose(pose_tuple: Tuple[np.ndarray, np.ndarray]) -> Pose:
    return Pose(
        position=Point(*pose_tuple[0]),
        orientation=Quaternion(*pose_tuple[1]),
    )


def _pose_to_tuple(pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ]
        ),
        np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        ),
    )


def initialize_moveit():
    robot_name = "locobot"
    robot_description = robot_name + "/robot_description"
    moveit_commander.roscpp_initialize(sys.argv)
    robot = RobotCommander(ns=robot_name, robot_description=robot_description)
    scene = PlanningSceneInterface(ns=robot_name)
    arm_group = MoveGroupCommander("interbotix_arm", ns=robot_name, robot_description=robot_description)
    return robot, scene, arm_group


class LocobotArm:
    """provide services to control the locobot arm (plan, plan cartesian and sleep)"""

    joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

    def __init__(self, serving=True) -> None:
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.robot, self.scene, self.arm_group = initialize_moveit()
        self.scale_factor = 0.8
        self.arm_group.set_max_velocity_scaling_factor(self.scale_factor)
        self.arm_group.set_max_acceleration_scaling_factor(self.scale_factor)

        self.arm_path_pub = rospy.Publisher("/locobot/arm/end_path", PoseArray, queue_size=5)

        self.ctl_state_sub = rospy.Subscriber(
            "/locobot/arm_controller/state", JointTrajectoryControllerState, self.on_ctl_state
        )

        if serving:
            ## reach goal poses (with any path)
            rospy.Service("/locobot/arm_control", SetPoseArray, self.reach)
            ## reach goal pose with cartesian path
            rospy.Service("locobot/arm_control_cartesian", SetPose, self.reach_c)
            ## config arm velocity and acceleration (scaling factor)
            rospy.Service("locobot/arm_config", SetFloat32, self.arm_config)
            ## for quick test
            rospy.Service("/locobot/arm_sleep", SetBool, self.sleep)
            ## for emergency stop
            rospy.Service("locobot/arm_stop", SetBool, self.stop)

    @property
    def end_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        ee_tf: TransformStamped = self.tf_buffer.lookup_transform(
            "locobot/arm_base_link", "locobot/ee_arm_link", rospy.Time()
        )
        return (
            np.array(
                [
                    ee_tf.transform.translation.x,
                    ee_tf.transform.translation.y,
                    ee_tf.transform.translation.z,
                ]
            ),
            np.array(
                [
                    ee_tf.transform.rotation.x,
                    ee_tf.transform.rotation.y,
                    ee_tf.transform.rotation.z,
                    ee_tf.transform.rotation.w,
                ]
            ),
        )

    def reach_c(self, req: SetPoseRequest):
        """reach goal pose with cartesian path"""
        target_position, target_oritation = _pose_to_tuple(req.data)
        success = self.move_to_pose_with_cartesian(position=target_position, rotation=target_oritation)
        resp = SetPoseResponse()
        resp.result = success
        resp.message = "success" if success else "failed"
        return resp

    def reach(self, req: SetPoseArrayRequest):
        """reach given waypoints in order"""
        errcode = self.move_to_poses(req.data)
        resp = SetPoseArrayResponse()
        resp.result = errcode == 0
        resp.message = "success" if errcode == 0 else ("plan failed" if errcode == 1 else "execution failed")
        return resp

    def print_traj(self, traj):
        for pnt in traj.joint_trajectory.points:
            pnt: JointTrajectoryPoint
            print(f"(time) {pnt.time_from_start.to_sec():.4f}\t(joints_value) ", end="")
            for i in pnt.positions:
                print(f"{i:.4f}, ", end="")
            print()
        return

    def move_to_poses(self, poses):
        """
        `poses`: list of Pose;
        return:
        `errcode`:
            0: success; 1: plan failed; 2: execution failed (just ignore it for now)
        """
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        stat0 = self.arm_group.get_current_state()
        trajs = []
        for i, pose in enumerate(poses):
            self.arm_group.set_start_state(stat0)
            self.arm_group.set_pose_target(pose)
            succ, traj, _, _ = self.arm_group.plan()
            traj: RobotTrajectory
            # print(f"{i}->{i+1}: {'success' if succ else 'fail'}")
            if not succ:
                return 1
            # self.print_traj(traj)
            trajs.append(traj)
            tmp = list(stat0.joint_state.position)
            tmp[2:8] = traj.joint_trajectory.points[-1].positions
            stat0.joint_state.position = tmp
        ## concatenate trajectories
        traj_concat = RobotTrajectory()
        traj_concat.joint_trajectory.joint_names = trajs[0].joint_trajectory.joint_names
        traj_concat.joint_trajectory.points = trajs[0].joint_trajectory.points
        for traj in trajs[1:]:
            traj_concat.joint_trajectory.points += traj.joint_trajectory.points[1:]
        traj_retime = self.arm_group.retime_trajectory(
            ref_state_in=self.arm_group.get_current_state(),
            traj_in=traj_concat,
            velocity_scaling_factor=self.scale_factor,
            acceleration_scaling_factor=self.scale_factor,
        )
        # print("retimed traj:")
        # self.print_traj(traj_retime)
        self.arm_group.execute(traj_retime)
        return 0 if np.max(self.joint_states_err) < 5e-2 else 2

    def arm_config(self, req: SetFloat32Request):
        factor = req.data
        resp = SetFloat32Response()
        if not (factor > 0 and factor <= 1):
            resp.result = False
            resp.message = f"provided scaling factor {factor:.3f} is not in (0, 1], current: {self.scale_factor:.3f}."
        else:
            self.arm_group.set_max_acceleration_scaling_factor(factor)
            self.arm_group.set_max_velocity_scaling_factor(factor)
            resp.result = True
            resp.message = (
                f"velocity and acceleration scaling factor changed from {self.scale_factor:.3f} to {factor:.3f}"
            )
            self.scale_factor = factor
        return resp

    def on_ctl_state(self, ctl_state: JointTrajectoryControllerState):
        self.joint_states = np.array(ctl_state.actual.positions)
        self.joint_states_goal = np.array(ctl_state.desired.positions)
        self.joint_states_err = np.array(ctl_state.error.positions)
        # rospy.loginfo(f"error: {self.joint_states_err}")

    def move_to_pose_with_cartesian(self, position, rotation, min_fraction=0.8):
        waypoint_poses = []
        waypoint_poses.append(_tuple_to_pose(self.end_pose))
        waypoint_poses.append(_tuple_to_pose((position, rotation)))
        path, fraction = self.arm_group.compute_cartesian_path(
            waypoints=waypoint_poses,
            eef_step=0.05,
        )
        print(f"Planned cartesian path with {len(path.joint_trajectory.points)} poses, fraction {fraction}")
        if fraction < min_fraction:
            print(f"Failed to plan cartesian path! The fraction {fraction} is below {min_fraction}.")
            return False
        if not self.arm_group.execute(path, wait=True):
            print("Failed to move arm along path!")
            return False
        print("Arm reached target!")
        return True

    def vis_pose_array(self, poses):
        ## waypoints
        wps = [_tuple_to_pose(pose) for pose in poses]
        self.arm_path_pub.publish(
            PoseArray(
                header=rospy.Header(
                    stamp=rospy.Time.now(),
                    frame_id="base_link",
                ),
                poses=wps,
            )
        )

    def move_cartesian_paths(
        self,
        waypoints_: List[List[Tuple[np.ndarray, np.ndarray]]],
        min_fraction=0.95,
        move_to_start_pose_first=False,
        fn_reach_start_pose: Optional[Callable[[], bool]] = None,
    ) -> bool:
        plan_success = False
        path: RobotTrajectory
        first_pose_trajectory: RobotTrajectory
        first_pose_joints: List[float]
        for waypoints in waypoints_:
            waypoint_poses = [_tuple_to_pose(pose) for pose in waypoints]
            self.arm_path_pub.publish(
                PoseArray(
                    header=rospy.Header(
                        stamp=rospy.Time.now(),
                        frame_id="base_link",
                    ),
                    poses=waypoint_poses,
                )
            )
            self.arm_group.clear_pose_targets()
            if move_to_start_pose_first:
                self.arm_group.set_start_state_to_current_state()
                self.arm_group.set_pose_target(waypoint_poses[0])
                first_pose_plan_success, first_pose_trajectory, _, _ = self.arm_group.plan()
                if not first_pose_plan_success:
                    continue
                first_pose_joints = first_pose_trajectory.joint_trajectory.points[-1].positions
                start_state = RobotState()
                start_state.joint_state.name = self.joint_names
                start_state.joint_state.position = first_pose_joints
                self.arm_group.set_start_state(start_state)
            else:
                self.arm_group.set_start_state_to_current_state()
            fraction: float
            path, fraction = self.arm_group.compute_cartesian_path(
                waypoints=waypoint_poses,
                eef_step=0.01,
                jump_threshold=1000.0,
            )
            print(f"Planned cartesian path with {len(path.joint_trajectory.points)} poses, fraction {fraction}")
            if fraction < min_fraction:
                print(f"Failed to plan cartesian path! The fraction {fraction} is below {min_fraction}.")
                continue
            plan_success = True
            break
        if not plan_success:
            return False

        if move_to_start_pose_first:
            print("Moving to first pose")
            if not self.arm_group.execute(first_pose_trajectory, wait=True):
                print("Failed to move to first pose!")
                return False
            print("Arm reached first pose!")
            if fn_reach_start_pose is not None:
                if not fn_reach_start_pose():
                    print("Start pose callback failed")
                    return False

        if not self.arm_group.execute(path, wait=True):
            print("Failed to move arm along path!")
            return False

        print("Arm reached target!")
        return True

    def move_joints(self, target_joints: np.ndarray) -> bool:
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        self.arm_group.set_start_state_to_current_state()
        self.arm_group.set_joint_value_target(target_joints)

        self.arm_group.go(wait=True)
        print(f"Move arm to joints {target_joints}")

    def sleep(self, msg: SetBoolRequest):
        if msg.data:
            self.move_joints(np.array([0.0, -1.1, 1.55, 0.0, 0.5, 0.0]))
            return SetBoolResponse(True, "locobot arm sleeped!")
        else:
            return SetBoolResponse(False, "Nothing to do.")

    def stop(self, msg: SetBoolRequest):
        if not msg.data:
            return SetBoolRequest(False, "invalid stop command.")
        self.arm_group.clear_pose_targets()
        self.arm_group.stop()
        return SetBoolResponse(True, "locobot arm stopped!")


if __name__ == "__main__":
    rospy.init_node("arm_controller")
    arm = LocobotArm(serving=False)
    p1 = Pose(
        position=Point(x=0.35, y=0, z=0.5),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
    )
    p2 = Pose(
        position=Point(x=0.4, y=0, z=0.4),
        orientation=Quaternion(x=0, y=0, z=0, w=1),
    )
    # import os
    # os.system('rosservice call /locobot/arm_control "data: {position: {x: 0.35, y: 0.0, z: 0.5}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}"')
    # os.system('rosservice call /locobot/arm_control "data: {position: {x: 0.4, y: 0.0, z: 0.4}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}"')
    arm.move_to_poses([p1])
    arm.move_to_poses([p2])
    print("done")
