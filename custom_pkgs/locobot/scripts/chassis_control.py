# /usr/bin/env python
from locobot.srv import SetPose2D, SetPose2DRequest, SetPose2DResponse
from move_base_msgs.msg import MoveBaseActionResult
from geometry_msgs.msg import *
import rospy
import tf2_ros
import tf.transformations

import numpy as np


class LocobotChassis:
    """provide services to control the chassis"""

    def __init__(self, serving=True):
        # state variables to access
        self.pose2d = None # (x, y, theta)

        # publish chassis goal
        self.pub_goal = rospy.Publisher("/locobot/move_base_simple/goal", PoseStamped, queue_size=1)
        # publish velocity command
        self.pub_vel = rospy.Publisher("/locobot/mobile_base/commands/velocity", Twist, queue_size=1)
        # publish chassis current 2D pose
        self.pub_curr = rospy.Publisher("/locobot/chassis/current_pose", Pose2D, queue_size=1)

        rospy.Subscriber("/locobot/move_base/result", MoveBaseActionResult, self.on_status)
        if serving:
            rospy.Service("/locobot/chassis_control", SetPose2D, self.on_chassis_control)

        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)

        rospy.Timer(rospy.Duration(0.05), self.publish_pose)
        while self.pose2d is None and not rospy.is_shutdown():
            rospy.sleep(0.05)
        rospy.loginfo("LocobotChassis initialized")

    def rotate(self, angle: float, max_angspd=1.2, max_duration:float=5.0):
        """PID control to rotate chassis by `angle` in rad, +: CCW, -: CW"""
        # PID parameters
        kp = 4
        ki = 1
        kd = 0.2

        dt = 0.05
        rate = rospy.Rate(1/dt)

        integral = 0
        last_error = 0
        cnt = 0
        rospy.logdebug(f"current angle: {self.pose2d[2]:.2f}, target angle: {self.pose2d[2] + angle:.2f}")
        angle_sp = self.pose2d[2] + angle
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown() and cnt < 5:
            if max_duration > 0 and rospy.Time.now().to_sec() - t0 > max_duration:
                raise TimeoutError(f"Rotate timeout")
            # calc error
            error = angle_sp - self.pose2d[2]
            error = np.arctan2(np.sin(error), np.cos(error))
            # update counter
            cnt = cnt + 1 if abs(error) < 0.1 else 0
            integral = integral + error * dt if abs(error) < 0.5 else 0 # integral separation
            derivative = (error - last_error) / dt
            # PID control output
            angspd_z = kp * error + ki * integral + kd * derivative
            # Limit angular speed
            angspd_z = np.clip(angspd_z, -max_angspd, max_angspd)
            # Create and publish Twist message
            twist_msg = Twist(Vector3(0, 0, 0), Vector3(0, 0, angspd_z))
            self.pub_vel.publish(twist_msg)
            # Update last error
            last_error = error
            rate.sleep()
        # Stop the robot after turning
        self.pub_vel.publish(Twist())
        rospy.logdebug(f"Rotation completed. Current angle: {self.pose2d[2]:.2f}")

    def on_chassis_control(self, req: SetPose2DRequest):
        self.reached = False
        quat = tf.transformations.quaternion_from_euler(0, 0, req.theta)
        # change pose2D to pose
        mes = PoseStamped()
        mes.header.stamp = rospy.Time.now()
        mes.header.frame_id = "map"
        mes.pose.position = Point(x=req.x, y=req.y, z=0.242)
        mes.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        # publish goal
        self.pub_goal.publish(mes)
        # wait for goal reached
        r = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.reached:
                break
            r.sleep()
        # return response
        resp = SetPose2DResponse()
        resp.result = True
        resp.message = "Goal reached"
        return resp

    def on_status(self, msg: MoveBaseActionResult):
        if msg.status.status == 3:
            self.reached = True

    def publish_pose(self, timer_event):
        msg = Pose2D()
        # look up `robot_base_frame` in "<locobot>/launch/move_base.launch"
        robot_base_frame = "locobot/base_footprint"
        map_frame = "map"
        if not self.tf_buf.can_transform(map_frame, robot_base_frame, rospy.Time(0)):
            rospy.logdebug(f"Cannot transform {map_frame} to {robot_base_frame}")
            return

        trans_stamped: TransformStamped = self.tf_buf.lookup_transform(map_frame, robot_base_frame, rospy.Time(0))
        quat = trans_stamped.transform.rotation
        msg.x = trans_stamped.transform.translation.x
        msg.y = trans_stamped.transform.translation.y
        msg.theta = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]
        self.pub_curr.publish(msg)
        self.pose2d = (msg.x, msg.y, msg.theta)


if __name__ == "__main__":
    rospy.init_node("chassis_controller")
    lc = LocobotChassis(serving=False)
    lc.rotate(-np.pi)
