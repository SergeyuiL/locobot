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
from std_msgs.msg import String

from move_base_msgs.msg import MoveBaseAction, MoveBaseResult, MoveBaseFeedback, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Pose, Twist
import actionlib
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse

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

    joint_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

    def __init__(self) -> None:

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.robot, self.scene, self.arm_group = initialize_moveit()
        # initialize_motor_pids()


        self.arm_waypoints_vis_pub = rospy.Publisher("/locobot/arm/end_path", PoseArray, queue_size=5)
        self._current_joint_state = rospy.wait_for_message("/locobot/joint_states", JointState)
        self._joint_state_sub = rospy.Subscriber("/locobot/joint_states", JointState, self._on_joint_state)
        self.tf_buffer.lookup_transform("locobot/arm_base_link", "locobot/ee_arm_link", rospy.Time(), rospy.Duration(5.0))

        self.arm_control_server = actionlib.SimpleActionServer("/locobot/arm_control", MoveBaseAction, self.exec_cb, auto_start=False)
        self.arm_control_server.start()

        self.arm_sleep_server = rospy.Service("/locobot/arm_sleep", SetBool, self.sleep_cb)

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
            self._current_joint_state.position[self._current_joint_state.name.index(joint_name)]
            for joint_name in self.joint_names
        ])
    
    def exec_cb(self, goal: MoveBaseGoal):
        target_position, target_oritation = _pose_to_tuple(goal.target_pose.pose)
        success = self.move_end_to_pose(position=target_position, orientation=target_oritation, nonblocking=False)
        if self.arm_control_server.preempt_request:
            rospy.loginfo("Cancel arm control ...")
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            res = PoseStamped()
            res.pose = _tuple_to_pose(self.end_pose)
            self.arm_control_server.set_preempted(result=res)
        else:
            if success:
                res = PoseStamped()
                res.pose = _tuple_to_pose(self.end_pose)
                self.arm_control_server.set_succeeded(result=res)
            else:
                res = PoseStamped()
                res.pose = _tuple_to_pose(self.end_pose)
                self.arm_control_server.set_aborted(result=res)

    def _on_joint_state(self, joint_state: JointState):
        self._current_joint_state = joint_state

    def move_end_to_pose(
            self,
            position: np.ndarray,
            orientation: np.ndarray,
            nonblocking=False
    ) -> bool:
        return self.move_end_to_poses([(position, orientation)], nonblocking)

    def move_end_to_poses(
            self,
            target_poses: List[Tuple[np.ndarray, np.ndarray]],
            nonblocking=False
    ) -> bool:
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        self.arm_group.set_start_state_to_current_state()
        self.arm_group.set_pose_targets([_tuple_to_pose(pose) for pose in target_poses])
        print(f"Moving arm to {target_poses}")
        start_time = rospy.get_time()
        plan_success, trajectory, planning_time, error_code = self.arm_group.plan()
        # print(trajectory)
        if not plan_success:
            print("Failed to move arm!")
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            return False
        if nonblocking:
            if self.arm_group.execute(trajectory, wait=False):
                return True
            else:
                print("Async execution failed")
                return False
        else:
            if not self.arm_group.execute(trajectory, wait=True):
                print("Execution failed")
                return False
            print("Arm reached target!")
            return True

    def vis_pose_array(self, poses):
        waypoint_poses = [_tuple_to_pose(pose) for pose in poses]
        self.arm_waypoints_vis_pub.publish(PoseArray(
            header=rospy.Header(
                stamp=rospy.Time.now(),
                frame_id="base_link",
            ),
            poses=waypoint_poses,
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
            self.arm_waypoints_vis_pub.publish(PoseArray(
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
                start_state.joint_state.name = self._current_joint_state.name
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
            nonblocking=False
    ) -> bool:
        self.arm_group.clear_pose_targets()
        self.arm_group.set_start_state_to_current_state()
        self.arm_group.set_joint_value_target(target_joints)
        print(f"Moving arm to joints {target_joints}")
        plan_success, trajectory, planning_time, error_code = self.arm_group.plan()
        if not plan_success:
            print("Failed to move arm!")
            self.arm_group.stop()
            return False

        if nonblocking:
            if self.arm_group.execute(trajectory, wait=False):
                return True
            else:
                print("Async execution failed")
                return False
        else:
            if not self.arm_group.execute(trajectory, wait=True):
                print("Execution failed")
                return False
            print("Arm reached target!")
            return True
    
    def sleep_cb(self, msg:SetBoolRequest):
        if msg.data:
            self.sleep()
            return SetBoolResponse(True, "locobot arm sleeped!")
        else:
            return SetBoolRequest(False, "Nothing todo.")
    def sleep(self):
        self.move_joints(np.array([0.0, -1.1, 1.55, 0.0, 0.5, 0.0]),nonblocking=True)

if __name__ == "__main__":
    rospy.init_node("locobot_arm_moveit_test")
    arm = LocobotArm()
    p1 = (np.array([0.5, -0.14, 0.15]), np.array([0.0, 0.0, 0.0, 1.0]))
    arm.move_end_to_pose(*p1)
    arm.sleep()
    rospy.spin()