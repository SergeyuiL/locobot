#!/usr/bin/env python3

import rospy
import sys
from odom_base_control.srv import BaseControl

def robot_control_client(command, value=0.0):
    """
    调用机器人控制服务
    
    参数:
        command: 命令类型 ("stop", "move_forward", "turn_left", "turn_right")
        value: 对应的值 (移动距离或旋转角度)
    """
    # 等待服务可用
    rospy.wait_for_service('base_control')
    
    try:
        # 创建服务代理
        control_service = rospy.ServiceProxy('base_control', BaseControl)
        
        # 调用服务
        response = control_service(command, value)
        
        # 返回结果
        return response.success, response.message
    
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return False, str(e)

def print_usage():
    """打印使用说明"""
    print("Usage:")
    print("  base_control_client.py stop")
    print("  base_control_client.py move_forward <distance_in_meters>")
    print("  base_control_client.py turn_left <angle_in_degrees>")
    print("  base_control_client.py turn_right <angle_in_degrees>")
    print("  Examples:")
    print("  base_control_client.py stop")
    print("  base_control_client.py move_forward 0.25")
    print("  base_control_client.py turn_left 30")
    print("  base_control_client.py turn_right 45")

if __name__ == "__main__":
    # 初始化ROS节点
    rospy.init_node('robot_control_client')
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    # 处理stop命令
    if command == "stop":
        success, message = robot_control_client("stop")
    
    # 处理其他命令
    elif command in ["move_forward", "turn_left", "turn_right"]:
        if len(sys.argv) < 3:
            print(f"Error: {command} command requires a value")
            print_usage()
            sys.exit(1)
        
        try:
            value = float(sys.argv[2])
            success, message = robot_control_client(command, value)
        except ValueError:
            print(f"Error: Invalid value '{sys.argv[2]}'. Must be a number.")
            sys.exit(1)
    
    # 未知命令
    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)
    
    # 打印结果
    if success:
        print(f"Success: {message}")
    else:
        print(f"Failed: {message}")
