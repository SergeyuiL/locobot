#!/usr/bin/env python3
import rospy
import tf

def listen_to_transform():
    rospy.init_node('car_base_link_listener')

    listener = tf.TransformListener()

    while not rospy.is_shutdown():
        input("Waiting for any input to print pose <-")  
        try:
            listener.waitForTransform("/map", "/locobot/lidar_tower_link", rospy.Time(0), rospy.Duration(4.0))
            (trans1, rot1) = listener.lookupTransform('/map', '/locobot/lidar_tower_link', rospy.Time(0))
            rospy.loginfo("/locobot/lidar_tower_link's position in /map frame:\n %s, %s" % (trans1, rot1))
            
            listener.waitForTransform("/locobot/camera_link", "/tag_5", rospy.Time(0), rospy.Duration(4.0))
            (trans2, rot2) = listener.lookupTransform('/locobot/camera_link', '/tag_5', rospy.Time(0))
            rospy.loginfo("/tag_5's position in /locobot/camera_link frame:\n %s, %s" % (trans2, rot2))
            
            listener.waitForTransform("/locobot/camera_color_optical_frame", "/locobot/camera_link", rospy.Time(0), rospy.Duration(4.0))
            (trans2, rot2) = listener.lookupTransform('/locobot/camera_color_optical_frame', '/locobot/camera_link', rospy.Time(0))
            rospy.loginfo("/locobot/camera_link's position in /locobot/camera_color_optical_frame frame:\n %s, %s" % (trans2, rot2))
            
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("TF Exception: %s" % str(e))

if __name__ == '__main__':
    listen_to_transform()