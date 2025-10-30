#!/usr/bin/env python3

## Provides a ros node class for pose estimation
## assuming you've got snapshots of the target (using scan.py)
## it reads in RGBD frames and publishs target's pose

import numpy as np
import open3d as o3d
import os
import cv2
import threading
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import sys

import rospy
import tf, tf2_ros
import tf.transformations
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion

from GMatch import gmatch


class PoseEstimator:
    def __init__(self, path_snapshots):
        rospy.init_node("pose_estimate", anonymous=True)
        with open(path_snapshots, 'rb') as f:
            snapshots = pickle.load(f)
        self.imgs_src, self.clds_src, self.masks_src, M_ex_list = zip(*snapshots)
        self.poses_src = [gmatch.util.mat2pose(M_ex) for M_ex in M_ex_list]

        self.lock_rgb = threading.Lock()
        self.lock_depth = threading.Lock()
        self.img_rgb = None
        self.img_dep = None
        
        cam_info: CameraInfo = rospy.wait_for_message(
            "/locobot/camera/color/camera_info", CameraInfo
        )
        self.cam_intrin = np.array(cam_info.K).reshape((3, 3))

        self.coord_targ = "pose_estimate/target"
        self.coord_cam = "locobot/camera_color_optical_frame"  ## not locobot/camera_aligned_depth_to_color_frame

        self.bridge = CvBridge()
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)
        self.tf_pub = tf2_ros.TransformBroadcaster()

        rospy.Subscriber("/locobot/camera/color/image_raw", Image, self.on_rec_img)
        rospy.Subscriber(
            "/locobot/camera/aligned_depth_to_color/image_raw", Image, self.on_rec_depth
        )

    def on_rec_img(self, msg):
        with self.lock_rgb:
            self.img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def on_rec_depth(self, msg):
        with self.lock_depth:
            self.img_dep = (
                self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1") * 1e-3
            )
    
    def run(self):
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            with self.lock_rgb:
                img_rgb = deepcopy(self.img_rgb)
            with self.lock_depth:
                img_dep = deepcopy(self.img_dep)
            if img_rgb is None or img_dep is None:
                rospy.logwarn("img_rgb or img_dep is not ready. keep waiting...")
                r.sleep()
                continue
            
            cv2.imshow("rgb", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imshow("depth", img_dep)
            cv2.waitKey(1)

            D_near = 1e-2
            D_far = 1
                
            match_data = gmatch.util.MatchData(
                imgs_src=self.imgs_src,
                clds_src=self.clds_src,
                masks_src=self.masks_src,
                poses_src=self.poses_src,
                img_dst=img_rgb,
                cld_dst=gmatch.util.depth2cld(img_dep, self.cam_intrin),
                mask_dst=np.where((img_dep > D_near) & (img_dep < D_far), 255, 0).astype(np.uint8),
            )

            t0 = rospy.Time.now()
            gmatch.Match(match_data, cache_id='default', debug=-1)
            gmatch.util.Solve(match_data)
            # gmatch.util.Refine(match_data)

            L = len(match_data.matches_list[match_data.idx_best])
            rospy.loginfo(f"gmatch costs {(rospy.Time.now() - t0)*1e-6} ms. match len: {L}")
            if L < 8:
                r.sleep()
                continue

            pos, rot = gmatch.util.mat2pose(match_data.mat_m2c)

            msg = TransformStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.coord_cam
            msg.child_frame_id = self.coord_targ
            msg.transform.translation = Vector3(*pos)
            msg.transform.rotation = Quaternion(*rot)

            self.tf_pub.sendTransform(msg)

            r.sleep()


if __name__ == '__main__':
    estim = PoseEstimator('/home/locobot/locobot/src/custom_pkgs/locobot/scripts/GMatch/cache/default_object.pt')
    estim.run()
