#!/usr/bin/env python3

import rospy
from arm_control import LocobotArm
from camera_control import LocobotCamera
from gripper_control import Gripper
from chassis_control import Chassis

if __name__ == "__main__":
    rospy.init_node("all_controller")
    gripper = Gripper()
    chassis = Chassis()
    arm = LocobotArm()
    camera = LocobotCamera()
    print("arm gripper camera and chassis are brought up.")
    rospy.spin()
