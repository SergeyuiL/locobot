#!/usr/bin/env python3

import rospy
from arm_control import LocobotArm
from camera_control import LocobotCamera
from gripper_control import Gripper

if __name__ == "__main__":
    rospy.init_node("all_controller")
    arm = LocobotArm()
    camera = LocobotCamera()
    gripper = Gripper()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        gripper.publish_state()
        rate.sleep()