# /usr/bin/env python
from locobot.srv import SetPose2D, SetPose2DRequest, SetPose2DResponse
from move_base_msgs.msg import MoveBaseActionResult
from geometry_msgs.msg import *
import rospy
import tf2_ros
import tf.transformations

import numpy as np


class PID:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.last_error = 0
        pass

    def ctl(self, error):
        derivative = (error - self.last_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output

    def feed(self, error):
        self.integral += error * self.dt
        self.last_error = error

    def reset(self):
        self.integral = 0
        self.last_error = 0


def remap_angle(angle: float, center=0.0) -> float:
    offset = center - np.pi
    return (angle - offset) % (2 * np.pi) + offset


class LocobotChassis:
    """provide services to control the chassis"""

    def __init__(self, serving=True):
        # state variables to access
        self.pose2d = None  # (x, y, theta)

        # publish chassis goal
        self.pub_goal = rospy.Publisher("/locobot/move_base_simple/goal", PoseStamped, queue_size=1)
        # publish velocity command
        self.pub_vel = rospy.Publisher("/locobot/mobile_base/commands/velocity", Twist, queue_size=1)
        # publish chassis current 2D pose
        self.pub_curr = rospy.Publisher("/locobot/chassis/current_pose", Pose2D, queue_size=1)

        if serving:
            rospy.Service("/locobot/chassis_control", SetPose2D, self.on_chassis_control)

        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)

        rospy.Timer(rospy.Duration(0.05), self.publish_pose)
        while self.pose2d is None and not rospy.is_shutdown():
            rospy.sleep(0.05)
        rospy.loginfo("LocobotChassis initialized")

    def rotate(self, delta_angle: float, max_angspd=1.2, max_duration=5.0):
        """PID control to rotate chassis by `delta_angle` in rad, +: CCW, -: CW
        Note: `delta_angle` will be remapped to [-pi, pi]
        """
        dt = 0.05
        pid = PID(kp=4.0, ki=1.0, kd=0.2, dt=0.05)
        rate = rospy.Rate(1 / dt)

        cnt = 0
        angle_sp = remap_angle(self.pose2d[2] + delta_angle)
        t0 = rospy.Time.now().to_sec()
        rospy.logdebug(f"current angle: {self.pose2d[2]:.2f}, target angle: {angle_sp:.2f}")
        while not rospy.is_shutdown() and cnt < 5:
            if max_duration > 0 and rospy.Time.now().to_sec() - t0 > max_duration:
                raise TimeoutError(f"Rotate timeout")

            # calc error
            error = remap_angle(angle_sp - self.pose2d[2])
            # PID calculations
            pid.feed(error)
            # integral separation
            if abs(error) >= 0.5:
                pid.integral = 0.0
            # clip PID control output
            angspd_z = np.clip(pid.ctl(error), -max_angspd, max_angspd)
            # publish Twist message
            self.pub_vel.publish(Twist(Vector3(0, 0, 0), Vector3(0, 0, angspd_z)))

            cnt = cnt + 1 if abs(error) < 0.1 else 0
            rate.sleep()
        # Stop the robot after turning
        self.pub_vel.publish(Twist())
        rospy.logdebug(f"Rotation completed. Current angle: {self.pose2d[2]:.2f}")

    def rotate_to(self, angle: float, max_angspd=1.2, max_duration=5.0):
        """PID control to rotate chassis to `angle` in rad w.r.t. map frame"""
        self.rotate(angle - self.pose2d[2], max_angspd, max_duration)

    def reach(self, x: float, y: float, theta=None, max_spd=0.3, max_angspd=1.2, max_duration=10.0):
        """PID control to reach `(x, y)` position w.r.t. map frame"""
        dt = 0.05
        pid_dist = PID(kp=2.0, ki=0.5, kd=0.1, dt=dt)
        pid_ang = PID(kp=4.0, ki=1.0, kd=0.2, dt=dt)
        rate = rospy.Rate(1 / dt)

        cnt = 0
        t0 = rospy.Time.now().to_sec()
        rospy.logdebug(
            f"current position: ({self.pose2d[0]:.2f}, {self.pose2d[1]:.2f}), target position: ({x:.2f}, {y:.2f})"
        )
        while not rospy.is_shutdown() and cnt < 5:
            if max_duration > 0 and rospy.Time.now().to_sec() - t0 > max_duration:
                raise TimeoutError(f"Reach position timeout")

            # calc distance and angle error
            dx = x - self.pose2d[0]
            dy = y - self.pose2d[1]
            angle_sp = np.arctan2(dy, dx)
            ang_error = remap_angle(angle_sp - self.pose2d[2])
            dist_error = np.sqrt(dx**2 + dy**2) * np.cos(ang_error)

            # PID calculations
            pid_ang.feed(ang_error)
            pid_dist.feed(dist_error)

            # integral separation
            if abs(ang_error) >= 0.5:
                pid_ang.integral = 0.0
            if abs(dist_error) >= 0.3:
                pid_dist.integral = 0.0

            # clip PID control output
            linspd_x = np.clip(pid_dist.ctl(dist_error), -max_spd, max_spd)
            angspd_z = np.clip(pid_ang.ctl(ang_error), -max_angspd, max_angspd)

            # only apply distance control when angle error is small
            if abs(ang_error) > 0.5:
                pid_dist.reset()
                linspd_x = 0.0

            # publish Twist message
            self.pub_vel.publish(Twist(Vector3(linspd_x, 0, 0), Vector3(0, 0, angspd_z)))

            cnt = cnt + 1 if np.sqrt(dx**2 + dy**2) < 0.05 else 0
            rate.sleep()
        if theta is not None:
            self.rotate_to(theta, max_angspd, max_duration=5.0)
        # Stop the robot after reaching
        self.pub_vel.publish(Twist())
        rospy.logdebug("Reached position: ({:.2f}, {:.2f}, {:.2f})".format(*self.pose2d))

    def on_chassis_control(self, req: SetPose2DRequest):
        self.reach(req.x, req.y)
        self.rotate_to(req.theta)
        # return response
        resp = SetPose2DResponse()
        resp.result = True
        resp.message = "Goal reached, final pose: ({:.2f}, {:.2f}, {:.2f})".format(*self.pose2d)
        return resp

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
    lc.reach(0, 0)
    # lc.rotate(-np.pi)
