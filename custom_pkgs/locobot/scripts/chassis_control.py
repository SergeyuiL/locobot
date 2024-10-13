from locobot.srv import setgoal, setgoalRequest, setgoalResponse
from geometry_msgs.msg import Pose, PoseStamped
import rospy

class Chassis:
    """ provide services to control the chassis """
    def __init__(self):
        self.chassis_pub = rospy.Publisher("/locobot/move_base_simple/goal", PoseStamped, queue_size=10)

        rospy.Service("/locobot/chassis_control", setgoal, self.on_chassis_control)
    
    def on_chassis_control(self, req:setgoalRequest):
        mes = PoseStamped()
        mes.header.stamp = rospy.Time.now()
        mes.header.frame_id = "map"
        mes.pose = req.goal
        self.chassis_pub.publish(mes)
        resp = setgoalResponse()
        resp.res = True
        resp.response = "published"
        return resp