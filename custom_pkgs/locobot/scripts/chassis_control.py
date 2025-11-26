# /usr/bin/env python
from locobot.srv import SetPose2D, SetPose2DRequest, SetPose2DResponse
from move_base_msgs.msg import MoveBaseActionResult
from geometry_msgs.msg import Pose, PoseStamped, Pose2D, Point, Quaternion, TransformStamped
import rospy
import tf2_ros
import tf.transformations


class Chassis:
    """provide services to control the chassis"""

    def __init__(self, serving=True):
        ## publish chassis goal
        self.pub_goal = rospy.Publisher("/locobot/move_base_simple/goal", PoseStamped, queue_size=10)
        ## publish chassis current 2D pose
        self.pub_curr = rospy.Publisher("/locobot/chassis/current_pose", Pose2D, queue_size=10)
        rospy.Subscriber("/locobot/move_base/result", MoveBaseActionResult, self.on_status)
        if serving:
            rospy.Service("/locobot/chassis_control", SetPose2D, self.on_chassis_control)

        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)

        rospy.Timer(rospy.Duration(0.1), self.publish_pose)

    def on_chassis_control(self, req: SetPose2DRequest):
        self.reached = False
        quat = tf.transformations.quaternion_from_euler(0, 0, req.theta)
        ## change pose2D to pose
        mes = PoseStamped()
        mes.header.stamp = rospy.Time.now()
        mes.header.frame_id = "map"
        mes.pose.position = Point(x=req.x, y=req.y, z=0.242)
        mes.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        ## publish goal
        self.pub_goal.publish(mes)
        ## wait for goal reached
        r = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.reached:
                break
            r.sleep()
        ## return response
        resp = SetPose2DResponse()
        resp.result = True
        resp.message = "Goal reached"
        return resp

    def on_status(self, msg: MoveBaseActionResult):
        if msg.status.status == 3:
            self.reached = True

    def publish_pose(self, timer_event):
        msg = Pose2D()
        ## look up `robot_base_frame` in "<locobot>/launch/move_base.launch"
        robot_base_frame = "locobot/base_footprint"
        map_frame = "map"
        if self.tf_buf.can_transform(map_frame, robot_base_frame, rospy.Time(0)):
            trans_stamped: TransformStamped = self.tf_buf.lookup_transform(map_frame, robot_base_frame, rospy.Time(0))
            quat = trans_stamped.transform.rotation
            msg.x = trans_stamped.transform.translation.x
            msg.y = trans_stamped.transform.translation.y
            msg.theta = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]
            self.pub_curr.publish(msg)
        else:
            print(f"no tf trans between {robot_base_frame} and {map_frame}")


if __name__ == "__main__":
    rospy.init_node("chassis_controller")
    chas = Chassis(serving=True)
    rospy.spin()
