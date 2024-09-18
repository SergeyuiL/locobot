import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

# 初始化ROS节点
rospy.init_node('data_collection', anonymous=True)

# 定义全局变量
images = []
odometry_data = []
bridge = CvBridge()
current_image = None

# TF监听器
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

def image_callback(msg):
    global current_image
    # 转换ROS图像消息为OpenCV图像
    current_image = bridge.imgmsg_to_cv2(msg, "bgr8")

def collect_data():
    global images, odometry_data
    try:
        # 获取locobot/lidar_tower_link在map坐标系的位姿
        trans = tf_buffer.lookup_transform('map', 'locobot/lidar_tower_link', rospy.Time(0))
        odometry_data.append(trans.transform)
        images.append(current_image)
        print("Data collected!")
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Failed to collect data")

def save_data():
    np.save('images.npy', images)
    np.save('odometry_data.npy', odometry_data)
    print("Data saved to files!")

# 订阅相机图像话题
rospy.Subscriber("/locobot/camera/color/image_raw", Image, image_callback)

print("Press SPACE to collect data. Press 's' to save data and exit.")
def user_input_thread():
    import sys, select
    while not rospy.is_shutdown():
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            input_text = sys.stdin.read(1)
            if input_text == ' ':
                collect_data()
            elif input_text == 's':
                save_data()
                rospy.signal_shutdown('User requested shutdown.')

import threading
input_thread = threading.Thread(target=user_input_thread)
input_thread.start()

rospy.spin()
input_thread.join()
