import rospy
import cv2

import rosbag
from cv_bridge import CvBridge
import numpy as np
import os
from sensor_msgs.msg import Image
import tf
from tf_conversions import transformations
class RGBDRecorder:
    def __init__(self, image_topic="/locobot/camera/color/image_raw", 
                        depth_topic="/locobot/camera/aligned_depth_to_color/image_raw", 
                        save_dir="/home/locobot/dataset/7190417"):
        self.rgb_path = os.path.join(save_dir, "rgb")
        self.depth_path = os.path.join(save_dir, "depth")
        self.pose_path = os.path.join(save_dir, "pose")
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.pose_path, exist_ok=True)

        self.depth_text_handle = open(os.path.join(save_dir, "depth.txt"), "w")
        self.rgb_text_handle = open(os.path.join(save_dir, "rgb.txt"), "w")
        self.bridge = CvBridge()
        rgb_sub = rospy.Subscriber(image_topic, Image, self.__rgb_callback, queue_size=1)
        depth_sub = rospy.Subscriber(depth_topic, Image, self.__depth_callback, queue_size=1)
        self.tf_listener = tf.TransformListener()
    

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

        tf_pose = self.__get_tf_pose(time=msg.header.stamp)
        if tf_pose is not None:
            pose_txt = "{} {} {} {} {} {} {}/n".format(tf_pose[0][0], tf_pose[0][1], tf_pose[0][2], 
                                                        tf_pose[1][0], tf_pose[1][1], tf_pose[1][2], tf_pose[1][3])
            txt_name = timestr + ".txt"
            with open(os.path.join(self.pose_path, txt_name), 'w') as f:
                f.write(pose_txt)
    
    def __get_tf_pose(self, time=None, frame="/map", childmap="/locobot/camera_tower_link"):
        if time is None:
            time = rospy.Time(0)
        try:
            (trans, rot) = self.tf_listener.lookupTransform(frame, childmap, time)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.loginfo("tf Error!")
            return None
        # rot_max = transformations.quaternion_matrix(rot)
        return trans, rot
        
    def close(self):
        self.depth_text_handle.close()
        self.rgb_text_handle.close()

if __name__ == "__main__":
    rospy.init_node("image_recoder")
    rgb_recoder = RGBDRecorder()
    rospy.spin()
    rgb_recoder.close()