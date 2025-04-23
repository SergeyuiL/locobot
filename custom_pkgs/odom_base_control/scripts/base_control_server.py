#!/usr/bin/env python3

import rospy
import math
import threading
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from odom_base_control.srv import BaseControl, BaseControlResponse

class PDController:
    """简单的PD控制器实现"""
    def __init__(self, kp, kd, output_min, output_max):
        self.kp = kp  # 比例增益
        self.kd = kd  # 微分增益
        self.output_min = output_min  # 输出下限
        self.output_max = output_max  # 输出上限
        
        self.reset()
        
    def reset(self):
        """重置控制器状态"""
        self.last_error = 0.0
        self.last_time = time.time()
        
    def compute(self, setpoint, process_variable):
        """计算PD输出"""
        # 计算时间差
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.01  # 防止除零错误
            
        # 计算误差
        error = setpoint - process_variable
        
        # 计算微分项
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        # 计算PD输出
        output = (self.kp * error) + (self.kd * derivative)
        
        # 限制输出范围
        output = max(self.output_min, min(self.output_max, output))
        
        # 更新状态
        self.last_error = error
        self.last_time = current_time
        
        return output

class RobotControlServer:
    def __init__(self):
        rospy.init_node('base_control_server')
        self.control_rate = 50
        
        # 机器人状态
        self.current_pose = None
        self.is_moving = False
        self.request_lock = threading.Lock()  # 用于请求队列锁
        
        # 发布者和订阅者
        self.cmd_vel_pub = rospy.Publisher('/locobot/mobile_base/commands/velocity', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/t265/odom/sample', Odometry, self.odom_callback)
        
        # 服务
        self.service = rospy.Service('base_control', BaseControl, self.handle_control_request)
        
        # 配置参数
        self.max_linear_speed = 1.0  # m/s
        self.max_angular_speed = 2.4  # rad/s
        self.position_tolerance = 0.02
        self.angle_tolerance = 0.01  
        self.default_timeout = 30.0  # 默认超时时间（秒）
        
        # 创建PD控制器
        # 线性移动PD
        self.linear_pd = PDController(
            kp=3.2,   # 比例增益
            kd=0.1,   # 微分增益
            output_min=-self.max_linear_speed,
            output_max=self.max_linear_speed
        )
        
        # 角度控制PD
        self.angular_pd = PDController(
            kp=7.4,   # 比例增益
            kd=0.1,   # 微分增益
            output_min=-self.max_angular_speed,
            output_max=self.max_angular_speed
        )
        
        rospy.loginfo("Robot Control Server initialized. Waiting for odometry data...")
        
        # 等待第一个里程计数据
        while self.current_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        rospy.loginfo("Odometry data received. Server ready.")
        
    def odom_callback(self, msg):
        """处理里程计数据"""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        # 从四元数获取欧拉角
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        # 更新当前位姿
        class Pose:
            pass
        
        self.current_pose = Pose()
        self.current_pose.x = position.x
        self.current_pose.y = position.y
        self.current_pose.theta = yaw
        
    def publish_cmd_vel(self, linear_x=0.0, angular_z=0.0):
        """发布速度命令"""
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)
        
    def stop(self):
        """停止机器人"""
        self.publish_cmd_vel(0.0, 0.0)
        self.is_moving = False
        # 重置所有PD控制器
        self.linear_pd.reset()
        self.angular_pd.reset()
        rospy.loginfo("Robot stopped")
        
    def handle_control_request(self, req):
        """处理控制请求，使用锁确保一次只处理一个请求"""
        # 尝试获取锁，如果已被占用，则等待（阻塞）
        with self.request_lock:
            response = BaseControlResponse()
            
            # 检查里程计数据是否可用
            if self.current_pose is None:
                response.success = False
                response.message = "No odometry data available!"
                return response
                
            try:
                if req.command == "stop":
                    self.stop()
                    response.success = True
                    response.message = "Robot stopped successfully"
                
                elif req.command == "move_forward":
                    success, message = self.move_forward(req.value)
                    response.success = success
                    response.message = message
                
                elif req.command == "turn_left":
                    success, message = self.turn_left(req.value)
                    response.success = success
                    response.message = message
                
                elif req.command == "turn_right":
                    success, message = self.turn_right(req.value)
                    response.success = success
                    response.message = message
                
                else:
                    response.success = False
                    response.message = f"Unknown command: {req.command}"
            
            except Exception as e:
                response.success = False
                response.message = f"Error executing command: {str(e)}"
                rospy.logerr(f"Error in handle_control_request: {str(e)}")
                
            return response
            
    def move_forward(self, distance):
        """向前移动指定距离，使用PD控制，等待完成后返回"""
        # 计算目标位置
        start_x = self.current_pose.x
        start_y = self.current_pose.y
        start_theta = self.current_pose.theta
        target_x = start_x + distance * math.cos(start_theta)
        target_y = start_y + distance * math.sin(start_theta)
        
        rospy.loginfo(f"Moving {'forward' if distance > 0 else 'backward'} {abs(distance):.2f} meters")
        
        # 重置PD控制器
        self.linear_pd.reset()
        
        # 设置移动状态
        self.is_moving = True
        
        # 计算超时时间（基于距离和速度，加上一些余量）
        timeout = abs(distance) / (self.max_linear_speed * 0.5) * 1.5 + 5.0
        if timeout > self.default_timeout:
            timeout = self.default_timeout
            
        # 等待直到达到目标位置或超时
        rate = rospy.Rate(self.control_rate)  
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            # 计算当前到目标的直线距离
            current_distance = math.sqrt(
                (self.current_pose.x - target_x)**2 + 
                (self.current_pose.y - target_y)**2
            )
            
            # 检查是否到达目标
            if current_distance < self.position_tolerance:
                self.stop()
                return True, f"Moved {'forward' if distance > 0 else 'backward'} {abs(distance):.2f} meters successfully"
                
            # 检查是否超时
            elapsed_time = (rospy.Time.now() - start_time).to_sec()
            if elapsed_time > timeout:
                self.stop()
                return False, f"Move timed out after {elapsed_time:.1f} seconds"
            
            # 计算前进方向（目标点相对于当前位置的角度）
            # target_heading = math.atan2(target_y - self.current_pose.y, 
                                        # target_x - self.current_pose.x)
            
            # 计算当前朝向与目标方向的角度差
            # heading_error = self.normalize_angle(target_heading - self.current_pose.theta)
            
            # 使用PD计算线性速度
            linear_speed = self.linear_pd.compute(0, -current_distance)
            
            # 确保方向正确
            if distance < 0:
                linear_speed = -abs(linear_speed)
            else:
                linear_speed = abs(linear_speed)
                
            # 使用简单的比例控制计算角速度（用于保持方向）
            # angular_speed = heading_error * 2.0
            # angular_speed = max(-self.max_angular_speed/2, min(self.max_angular_speed/2, angular_speed))
            
            # 发布速度命令
            self.publish_cmd_vel(linear_speed, 0.0)
            
            rospy.loginfo(f"Current position: ({self.current_pose.x:.2f}, {self.current_pose.y:.2f}), "
                          f"Target position: ({target_x:.2f}, {target_y:.2f}), "
                          f"Distance to target: {current_distance:.2f}, ")
            
            rate.sleep()

            
    def turn_left(self, angle_degrees):
        """向左转指定角度，等待完成后返回"""
        return self.turn(angle_degrees)
        
    def turn_right(self, angle_degrees):
        """向右转指定角度，等待完成后返回"""
        return self.turn(-angle_degrees)
        
    def turn(self, angle_degrees):
        """转动指定角度，正值为左转，负值为右转"""
        # 转换为弧度
        angle_radians = math.radians(angle_degrees)
        
        # 计算目标角度
        start_theta = self.current_pose.theta
        target_theta = start_theta + angle_radians
        
        # 规范化到[-pi, pi]
        target_theta = self.normalize_angle(target_theta)
        
        direction = "left" if angle_degrees > 0 else "right"
        rospy.loginfo(f"Turning {direction} {abs(angle_degrees):.1f} degrees from {math.degrees(start_theta):.1f} to {math.degrees(target_theta):.1f}")
        
        # 重置PD控制器
        self.angular_pd.reset()
        
        # 设置移动状态
        self.is_moving = True
        
        # 计算超时时间
        timeout = abs(angle_radians) / (self.max_angular_speed * 0.5) * 1.5 + 3.0
        if timeout > self.default_timeout:
            timeout = self.default_timeout
            
        # 等待直到达到目标角度或超时
        rate = rospy.Rate(self.control_rate)  
        start_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            # 计算当前角度与目标角度的差异
            current_theta = self.current_pose.theta
            angle_diff = self.normalize_angle(target_theta - current_theta)
            
            # 检查是否达到目标角度
            if abs(angle_diff) < self.angle_tolerance:
                self.stop()
                return True, f"Turned {direction} {abs(angle_degrees):.1f} degrees successfully"
                
            # 检查是否超时
            elapsed_time = (rospy.Time.now() - start_time).to_sec()
            if elapsed_time > timeout:
                self.stop()
                return False, f"Turn timed out after {elapsed_time:.1f} seconds"
                
            # 使用PD计算角速度
            angular_speed = self.angular_pd.compute(0, angle_diff)
            
            # 确保方向正确
            if angle_degrees < 0 and angular_speed > 0:  # 应该右转但计算结果为左转
                angular_speed = -angular_speed
            elif angle_degrees > 0 and angular_speed < 0:  # 应该左转但计算结果为右转
                angular_speed = -angular_speed
                
            # 发布速度命令
            self.publish_cmd_vel(0.0, angular_speed)
            
            # rospy.loginfo(f"Current angle: {math.degrees(current_theta):.1f}, Target angle: {math.degrees(target_theta):.1f}, Angle diff: {math.degrees(angle_diff):.1f}")
            
            rate.sleep()

            
    def normalize_angle(self, angle):
        """将角度规范化到[-pi, pi]范围内"""
        return math.atan2(math.sin(angle), math.cos(angle))
        
    def calculate_path_deviation(self, start_x, start_y, target_x, target_y):
        """计算机器人与直线路径的偏差"""
        if start_x == target_x and start_y == target_y:
            return 0.0
            
        # 计算直线方程 Ax + By + C = 0
        A = target_y - start_y
        B = start_x - target_x
        C = target_x * start_y - start_x * target_y
        
        # 计算点到直线的距离
        numerator = abs(A * self.current_pose.x + B * self.current_pose.y + C)
        denominator = math.sqrt(A**2 + B**2)
        
        return numerator / denominator

if __name__ == "__main__":
    try:
        server = RobotControlServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
