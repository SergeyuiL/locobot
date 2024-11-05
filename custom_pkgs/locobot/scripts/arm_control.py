#!/usr/bin/env python3

from typing import Callable, List, Optional, Tuple
import threading

from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, TransformStamped
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import RobotTrajectory, RobotState
from sensor_msgs.msg import JointState
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

def _tuple_to_pose(pose_tuple: Tuple[np.ndarray, np.ndarray]) -> Pose:
    return Pose(
        position=Point(
            x=pose_tuple[0][0],
            y=pose_tuple[0][1],
            z=pose_tuple[0][2],
        ),
        orientation=Quaternion(
            x=pose_tuple[1][0],
            y=pose_tuple[1][1],
            z=pose_tuple[1][2],
            w=pose_tuple[1][3],
        ),
    )

def _pose_to_tuple(pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ]),
        np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ])
    )

def initialize_moveit():
    robot_name = "locobot"
    robot_description = robot_name + "/robot_description"
    moveit_commander.roscpp_initialize(sys.argv)
    robot = RobotCommander(ns=robot_name, robot_description=robot_description)
    scene = PlanningSceneInterface(ns=robot_name)
    arm_group = MoveGroupCommander("interbotix_arm", ns=robot_name, robot_description=robot_description)
    return robot, scene, arm_group

class LocobotArm():
    """ provide services to control the locobot arm (plan, plan cartesian and sleep) """
    jnts_name = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

    def __init__(self) -> None:

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.robot, self.scene, self.arm_group = initialize_moveit()
        self.arm_group.set_max_acceleration_scaling_factor(0.5)
        self.arm_group.set_max_velocity_scaling_factor(0.5)
        # initialize_motor_pids()

        self.arm_path_pub = rospy.Publisher("/locobot/arm/end_path", PoseArray, queue_size=5)

        self.jnt_stats = rospy.wait_for_message("/locobot/joint_states", JointState)   # current joint states
        self.jnt_stats_sub = rospy.Subscriber("/locobot/joint_states", JointState, self.on_jnt_stats)
        self.tf_buffer.lookup_transform("locobot/arm_base_link", "locobot/ee_arm_link", rospy.Time(), rospy.Duration(5.0))

        ## reach goal pose (with any path)
        rospy.Service("/locobot/arm_control", SetPose, self.reach)
        
        ## reach goal pose with cartesian path
        rospy.Service("locobot/arm_control_cartesian", SetPose, self.reach_c)

        rospy.Service("locobot/arm_config", SetFloat32, self.arm_config)

        ## for quick test
        rospy.Service("/locobot/arm_sleep", SetBool, self.sleep)

    @property
    def end_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        ee_tf: TransformStamped = self.tf_buffer.lookup_transform("locobot/arm_base_link", "locobot/ee_arm_link", rospy.Time())
        return (np.array([
            ee_tf.transform.translation.x,
            ee_tf.transform.translation.y,
            ee_tf.transform.translation.z,
        ]), np.array([
            ee_tf.transform.rotation.x,
            ee_tf.transform.rotation.y,
            ee_tf.transform.rotation.z,
            ee_tf.transform.rotation.w,
        ]))

    @property
    def joint_values(self) -> np.ndarray:
        return np.array([
            self._jnt_stats.position[self._jnt_stats.name.index(joint_name)]
            for joint_name in self.jnts_name
        ])
    
    def reach(self, req:SetPoseRequest):
        """ reach goal popse with any path """
        print(">>>>>>>>>>>>>>>>>>>>")
        resp = SetPoseResponse()
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        rospy.sleep(2)
        self.arm_group.set_start_state_to_current_state()
        self.arm_group.set_pose_target(req.data)
        success, trajectory, plan_time, error_code = self.arm_group.plan()
        print(f"plan_success: {success}, plan_time: {plan_time:.3f}, error_code: {error_code}")
        if not success:
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            resp.result = False
            resp.message = "Failed in Planning"
        else:
            ## Somehow, last failure in execution will stop execution this time.
            ## therefore, call go() twice to make sure it to executes.
            self.arm_group.go(wait=True)
            self.arm_group.go(wait=True)
            resp.result = True
            resp.message = "Succeed"
        print("<<<<<<<<<<<<<<<<<<<<\n")
        return resp
    
    def reach_c(self, req:SetPoseRequest):
        """ reach goal pose with cartesian path """
        target_position, target_oritation = _pose_to_tuple(req.data)
        success = self.move_to_pose_with_cartesian(position=target_position, rotation=target_oritation)
        resp = SetPoseResponse()
        resp.result = success
        resp.message = "success" if success else "failed"
        return resp

    def arm_config(self, req:SetFloat32Request):
        factor = req.data
        resp = SetFloat32Response()
        if not (factor > 0 and factor <= 1):
            resp.result = False
            resp.message = f"provided scaling factor {factor:.3f} is not in (0, 1]"
        else:
            self.arm_group.set_max_acceleration_scaling_factor(factor)
            self.arm_group.set_max_velocity_scaling_factor(factor)
            resp.result = True
            resp.message = f"velocity and acceleration scaling factor changed to {factor:.3f}"
        return resp

    def on_jnt_stats(self, joint_state: JointState):
        self._jnt_stats = joint_state
    
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
        self.arm_path_pub.publish(PoseArray(
            header=rospy.Header(
                stamp=rospy.Time.now(),
                frame_id="base_link",
            ),
            poses=wps,
        ))

    def move_cartesian_paths(
            self,
            waypoints_: List[List[Tuple[np.ndarray, np.ndarray]]],
            min_fraction=0.95,
            move_to_start_pose_first=False,
            fn_reach_start_pose: Optional[Callable[[],bool]]=None,
        ) -> bool:
        plan_success = False
        path: RobotTrajectory
        first_pose_trajectory: RobotTrajectory
        first_pose_joints: List[float]
        for waypoints in waypoints_:
            waypoint_poses = [_tuple_to_pose(pose) for pose in waypoints]
            self.arm_path_pub.publish(PoseArray(
                header=rospy.Header(
                    stamp=rospy.Time.now(),
                    frame_id="base_link",
                ),
                poses=waypoint_poses,
            ))
            self.arm_group.clear_pose_targets()
            if move_to_start_pose_first:
                self.arm_group.set_start_state_to_current_state()
                self.arm_group.set_pose_target(waypoint_poses[0])
                first_pose_plan_success, first_pose_trajectory, _, _ = self.arm_group.plan()
                if not first_pose_plan_success:
                    continue
                first_pose_joints = first_pose_trajectory.joint_trajectory.points[-1].positions
                start_state = RobotState()
                start_state.joint_state.name = self._jnt_stats.name
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

    def move_joints(
            self,
            target_joints: np.ndarray,
    ) -> bool:
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        self.arm_group.set_start_state_to_current_state()
        self.arm_group.set_joint_value_target(target_joints)

        self.arm_group.go(wait=True)
        self.arm_group.go(wait=True)
        # print(f"Move arm to joints {target_joints}")
    
    def sleep(self, msg:SetBoolRequest):
        if msg.data:
            self.move_joints(np.array([0.0, -1.1, 1.55, 0.0, 0.5, 0.0]))
            return SetBoolResponse(True, "locobot arm sleeped!")
        else:
            return SetBoolResponse(False, "Nothing to do.")

if __name__ == "__main__":
    rospy.init_node("locobot_arm_moveit_test")
    arm = LocobotArm()
    # p1 = (np.array([0.5, -0.14, 0.15]), np.array([0.0, 0.0, 0.0, 1.0]))
    # arm.move_end_to_pose(*p1)
    # arm.move_to_pose_with_cartesian(np.array([0.45, 0, 0.45]), np.array([0.0, 0.0, 0.0, 1.0]))
    # arm.sleep()
    rospy.spin()