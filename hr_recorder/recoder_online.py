import rospy
import cv2

import rosbag
from cv_bridge import CvBridge
import numpy as np
import os
from sensor_msgs.msg import Image

class RGBDRecorder:
    def __init__(self, image_topic="/locobot/camera/color/image_raw", 
                        depth_topic="/locobot/camera/aligned_depth_to_color/image_raw", 
                        save_dir="/home/locobot/locobot/src/hr_recorder/7190408"):
        self.rgb_path = os.path.join(save_dir, "rgb")
        self.depth_path = os.path.join(save_dir, "depth")
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)

        self.depth_text_handle = open(os.path.join(save_dir, "depth.txt"), "w")
        self.rgb_text_handle = open(os.path.join(save_dir, "rgb.txt"), "w")
        self.bridge = CvBridge()
        rgb_sub = rospy.Subscriber(image_topic, Image, self.__rgb_callback, queue_size=1)
        depth_sub = rospy.Subscriber(depth_topic, Image, self.__depth_callback, queue_size=1)
    

    def __rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        timestr = "%.6f" % msg.header.stamp.to_sec()
        image_name = timestr + ".png"
        image_path = "rgb/" + image_name
        self.rgb_text_handle.write(timestr + " " + image_path + "\n")
        cv2.imwrite(os.path.join(self.rgb_path, image_name), cv_image)
    
    def __depth_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        timestr = "%.6f" % msg.header.stamp.to_sec()
        image_name = timestr + ".png"
        image_path = "depth/" + image_name
        self.depth_text_handle.write(timestr + " " + image_path + "\n")
        cv2.imwrite(os.path.join(self.depth_path, image_name), cv_image)
    
    def close(self):
        self.depth_text_handle.close()
        self.rgb_text_handle.close()

if __name__ == "__main__":
    rospy.init_node("image_recoder")
    rgb_recoder = RGBDRecorder()
    rospy.spin()
    rgb_recoder.close()