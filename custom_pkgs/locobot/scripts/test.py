#!/usr/bin/env python3
## used to test whether locobot arm, chassis and camera work fine

import rospy
from locobot.srv import setgoal, setgoalRequest, setgoalResponse
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse

def test_arm():
    arm_proxy = rospy.ServiceProxy("/locobot/arm_control", setgoal)
    arm_proxy.wait_for_service()
    req = setgoalRequest()
    req.goal.position.x = 0.5
    req.goal.position.y = 0
    req.goal.position.z = 0.3
    req.goal.orientation.x = 0
    req.goal.orientation.y = 0
    req.goal.orientation.z = 0
    req.goal.orientation.w = 1
    arm_proxy(req)

def test_gripper():
    gripper_proxy = rospy.ServiceProxy("/locobot/gripper_control", SetBool)
    gripper_proxy.wait_for_service()
    req = SetBoolRequest()
    req.data = False
    gripper_proxy(req)
    req.data = True
    gripper_proxy(req)

def test_chassis():
    chassis_proxy = rospy.ServiceProxy("/locobot/chassis_control", setgoal)
    chassis_proxy.wait_for_service()
    req = setgoalRequest()
    req.goal.position.x = 0
    req.goal.position.y = 0
    req.goal.position.z = 0
    req.goal.orientation.x = 0
    req.goal.orientation.y = 0
    req.goal.orientation.z = 0
    req.goal.orientation.w = 1
    chassis_proxy(req)

if __name__ == "__main__":
    test_arm()
    test_chassis()
    test_gripper()