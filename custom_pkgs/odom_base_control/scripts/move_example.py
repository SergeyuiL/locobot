#!/usr/bin/env python3

import rospy
from odom_base_control.srv import BaseControl

def execute_command(control_service, command, value):
    """执行单个命令并等待完成"""
    print(f"Executing command: {command} {value}")
    
    try:
        # 调用服务并等待响应（阻塞）
        response = control_service(command, value)
        
        # 输出结果
        if response.success:
            print(f"✓ Success: {response.message}")
        else:
            print(f"✗ Failed: {response.message}")
            
        rospy.sleep(0.5)
            
        return response.success
        
    except rospy.ServiceException as e:
        print(f"✗ Service call failed: {e}")
        return False

def execute_sequence():
    """执行一系列机器人动作"""
    rospy.init_node('robot_control_client')
    
    print("Waiting for base_control service...")
    rospy.wait_for_service('base_control')
    control_service = rospy.ServiceProxy('base_control', BaseControl)
    print("Service found! Starting command sequence...")
    
    try:
        # # 执行12次左转
        # for i in range(12):
        #     print(f"--- Turn left {i+1}/12 ---")
        #     if not execute_command(control_service, "turn_left", 30.0):
        #         print("Sequence aborted due to command failure")
        #         return
        
        # # 前进
        # print("--- Moving forward ---")
        # if not execute_command(control_service, "move_forward", 0.25):
        #     print("Sequence aborted due to command failure")
        #     return
        
        # # 右转
        # print("--- Turning right ---")
        # if not execute_command(control_service, "turn_right", 30.0):
        #     print("Sequence aborted due to command failure")
        #     return
        
        # 前进
        print("--- Moving forward ---")
        if not execute_command(control_service, "move_forward", 0.25):
            print("Sequence aborted due to command failure")
            return
        
        # 右转
        # print("\n--- Turning right ---")
        # if not execute_command(control_service, "turn_right", 30.0):
        #     print("Sequence aborted due to command failure")
        #     return
        
        # 左转
        # print("\n--- Turning left ---")
        # if not execute_command(control_service, "turn_left", 30.0):
        #     print("Sequence aborted due to command failure")
        #     return
        
        print("✓✓✓ Sequence completed successfully! ✓✓✓")
        
    except KeyboardInterrupt:
        print("Sequence interrupted by user")
    except Exception as e:
        print(f"Error during sequence execution: {e}")

if __name__ == "__main__":
    try:
        execute_sequence()
    except rospy.ROSInterruptException:
        pass
