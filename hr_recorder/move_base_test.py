import rospy
import actionlib

from move_base_msgs.msg import *
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import *
from tf_conversions import transformations

def go_to_pose(x, y, theta):
    sac = actionlib.SimpleActionClient("/locobot/move_base", MoveBaseAction)

    goal = MoveBaseGoal()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    q = transformations.quaternion_from_euler(0.0,0.0, theta)
    goal.target_pose.pose.orientation.x = q[0]
    goal.target_pose.pose.orientation.y = q[1]
    goal.target_pose.pose.orientation.z = q[2]
    goal.target_pose.pose.orientation.w = q[3]

    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    
    sac.wait_for_server()
    sac.send_goal(goal)
    print("Sending goal: ", x, y, theta)
    sac.wait_for_result()
    print(sac.get_result())


if __name__ == '__main__':
    try:
        rospy.init_node("test_move_base")
        go_to_pose(0.8, 0.8, 0.0)
    except rospy.ROSInternalException:
        print("something wrong!")