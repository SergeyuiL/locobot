#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
from interbotix_xs_modules.arm import InterbotixRobotXSCore, InterbotixGripperXSInterface
from interbotix_xs_msgs.srv import MotorGains, MotorGainsRequest
from interbotix_xs_msgs.srv import RegisterValues, RegisterValuesRequest
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse

def initialize_gripper():
    dxl = InterbotixRobotXSCore(
        robot_model="wx250s",
        robot_name="locobot",
        init_node=False,
    )
    dxl.robot_set_operating_modes("single", "gripper", "pwm")
    dxl.robot_torque_enable("single", "gripper", True)
    gripper = InterbotixGripperXSInterface(
        core=dxl,
        gripper_name="gripper",
        gripper_pressure=0.5,
        gripper_pressure_lower_limit=150,
        gripper_pressure_upper_limit=350
    )
    return dxl, gripper

srv_set_gain_name = "locobot/set_motor_pid_gains"
srv_set_gain = rospy.ServiceProxy(srv_set_gain_name, MotorGains)

srv_set_reg_name = "locobot/set_motor_registers"
srv_set_reg = rospy.ServiceProxy(srv_set_reg_name, RegisterValues)

def set_gain(motor:str, ki_vel: int, kp_vel:int, kd_pos: int, ki_pos: int, kp_pos: int, ff_gain1: int, ff_gain2: int):
    rospy.wait_for_service(srv_set_gain_name)
    print(f"Setting {motor} PID to {ki_vel}, {kp_vel}, {kd_pos}, {ki_pos}, {kp_pos}")
    req = MotorGainsRequest()
    req.cmd_type = "single"
    req.name = motor
    req.ki_vel = ki_vel
    req.kp_vel = kp_vel
    req.kd_pos = kd_pos
    req.ki_pos = ki_pos
    req.kp_pos = kp_pos
    req.k1 = ff_gain1
    req.k2 = ff_gain2
    srv_set_gain.call(req)

def set_reg(motor:str, reg:str, val:int):
    rospy.wait_for_service(srv_set_reg_name)
    reg_req = RegisterValuesRequest()
    reg_req.cmd_type = "single"
    reg_req.name = motor
    reg_req.reg = reg
    reg_req.value = val
    srv_set_reg.call(reg_req)

def initialize_motor_pids():
    set_gain("waist", 1920, 100, 0, 1000, 2400, 500, 50)
    set_gain("shoulder", 1920, 100, 3000, 400, 800, 500, 50)
    set_gain("elbow", 1920, 100, 0, 1600, 1200, 800, 0)
    set_gain("forearm_roll", 1920, 100, 0, 500, 2000, 500, 0)
    set_gain("wrist_angle", 1920, 100, 0, 100, 2000, 0, 0)
    set_gain("wrist_rotate", 1000, 100, 3600, 0, 1000, 0, 0)
    set_reg("waist", "Profile_Acceleration", 6)
    set_reg("shoulder", "Profile_Acceleration", 6)
    set_reg("elbow", "Profile_Acceleration", 6)
    set_reg("forearm_roll", "Profile_Acceleration", 6)
    set_reg("wrist_angle", "Profile_Acceleration", 6)
    set_reg("wrist_rotate", "Profile_Acceleration", 6)

class Gripper:
    """ publish gripper state (close or not) and provide service to control gripper """
    def __init__(self):
        print("-------------init moto pids---------------")
        initialize_motor_pids()
        print("----------------init gripper----------------")
        self.dxl, self.gripper_impl = initialize_gripper()
        print("---------------------complete init---------------------")
        self.gripper_state = "close"
        self.close()
        self.gripper_state_pub = rospy.Publisher("/locobot/gripper_state", String, queue_size=20)
        # self.gripper_contorl_sub = rospy.Subscriber("/locobot/gripper_control", String, self.control_callback)
        self.gripper_service = rospy.Service("/locobot/gripper_control", SetBool, self.close_gripper)
    
    def close_gripper(self, req: SetBoolRequest):
        if req.data:
            rospy.loginfo("close gripper ...")
            self.close()
            return SetBoolResponse(True, "closed girpper")
        else:
            rospy.loginfo("open gripper ...")
            self.open()
            return SetBoolResponse(True, "opened gripper")
        
    # def control_callback(self, msg):
    #     if msg.data == "open":
    #         self.open()
    #     elif msg.data == "close":
    #         self.close()

    def open(self):
        self.gripper_impl.open(delay=2.0)
        self.gripper_state = "open"

    def close(self):
        self.gripper_impl.close(delay=2.0)
        self.gripper_state = "close"

    @property
    def state(self):
        if self.gripper_impl.gripper_moving:
            return "moving"
        elif self.gripper_impl.gripper_command.cmd < 0:
            return "close"
        elif self.gripper_impl.gripper_command.cmd > 0:
            return "open"
        else:
            # print(self.gripper_impl.gripper_command.cmd)
            return self.gripper_state
    def publish_state(self):
        state = self.state
        self.gripper_state_pub.publish(state)

if __name__ == "__main__":
    rospy.init_node("gripper_controller")
    gripper = Gripper()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        gripper.publish_state()
        rate.sleep()
