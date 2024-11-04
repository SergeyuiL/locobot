from locobot.srv import SetPose, SetPoseRequest, SetPoseResponse
from move_base_msgs.msg import MoveBaseActionResult
from geometry_msgs.msg import Pose, PoseStamped
import rospy

class Chassis:
    """ provide services to control the chassis """
    def __init__(self):
        self.chassis_pub = rospy.Publisher("/locobot/move_base_simple/goal", PoseStamped, queue_size=10)
        rospy.Subscriber("/locobot/move_base/result", MoveBaseActionResult, self.on_status)
        rospy.Service("/locobot/chassis_control", SetPose, self.on_chassis_control)
    
    def on_chassis_control(self, req:SetPoseRequest):
        self.reached = False
        ## publish goal
        mes = PoseStamped()
        mes.header.stamp = rospy.Time.now()
        mes.header.frame_id = "map"
        mes.pose = req.data
        self.chassis_pub.publish(mes)
        ## wait for goal reached
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.reached:
                break
            r.sleep()
        ## return response
        resp = SetPoseResponse()
        resp.result = True
        resp.message = "Goal reached"
        return resp

    def on_status(self, msg:MoveBaseActionResult):
        if msg.status.status == 3:
            self.reached = True