#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import os
import cv2
import threading
from copy import deepcopy

try:
    import rospy
    import tf, tf2_ros
    import tf.transformations
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CameraInfo, Image
except:
    pass


def reconstruct(depth: np.ndarray, cam_intrin: np.ndarray):
    """reconstruct point cloud from depth image"""
    if len(depth) < 2:
        return None
    w, h = depth.shape[-2:]
    if w == 0 or h == 0:
        return None
    u, v = np.meshgrid(np.arange(h), np.arange(w), indexing="xy")
    z = depth
    x = (u - cam_intrin[0, 2]) * z / cam_intrin[0, 0]
    y = (v - cam_intrin[1, 2]) * z / cam_intrin[1, 1]
    cld = np.stack([x, y, z], axis=-1)
    return cld


class ModelLib:
    def __init__(self, load_only):
        """load_only: if True, only load the model library without ROS dependencies. if `sample_wizard()` is needed, set it to False"""
        self.model_folder = os.path.join(os.path.dirname(__file__), "ModelLib")
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        self.load_only = load_only
        if not load_only:
            rospy.init_node("pose_estimate")
            self.lock_rgb = threading.Lock()
            self.lock_depth = threading.Lock()
            self.init_ros_vars()

    def init_ros_vars(self):
        cam_info: CameraInfo = rospy.wait_for_message(
            "/locobot/camera/color/camera_info", CameraInfo
        )
        self.cam_intrin = np.array(cam_info.K).reshape((3, 3))
        self.coord_map = "map"
        self.coord_cam = "locobot/camera_color_optical_frame"  ## not locobot/camera_aligned_depth_to_color_frame
        self.bridge = CvBridge()
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        rospy.Subscriber("/locobot/camera/color/image_raw", Image, self.on_rec_img)
        rospy.Subscriber(
            "/locobot/camera/aligned_depth_to_color/image_raw", Image, self.on_rec_depth
        )

    def load(self, model_name):
        """load snapshots of the model"""
        model_path = os.path.join(self.model_folder, model_name + ".npy")
        imgs_rgb, clds = np.load(model_path)
        return imgs_rgb, clds

    def visualize_snapshots(self, imgs_rgb, clds):
        """visualize the point cloud of the model"""
        print("visualizing the point cloud")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.reshape(clds, (-1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.reshape(imgs_rgb, (-1, 3)) / 255.0)
        o3d.visualization.draw_geometries([pcd])
        # visualize the point cloud with color

    def sample_wizard(self, model_name):
        """sample a few snapshots of the model with given name"""
        model_path = os.path.join(self.model_folder, model_name + ".npy")
        print("Press 's' to save the current snapshot")
        print("Press 'q' to quit")
        snapshots = []  # list of (rgb, depth, pose)
        while True:
            if hasattr(self, "img_rgb"):
                bgr = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("camera", bgr)
                key = cv2.waitKey(33)
            else:
                rospy.sleep(0.1)
                continue
            if key == ord("s"):
                print("saving snapshot")
                snapshots.append(self.snapshot())
            elif key == ord("q"):
                print("quitting")
                break
        imgs_rgb, clds = [], []
        for i, snap in enumerate(snapshots):
            img_rgb, img_dep, pose = snap
            imgs_rgb.append(img_rgb)
            cld_cam = reconstruct(img_dep, self.cam_intrin)  # (h, w, 3)
            R = tf.transformations.quaternion_matrix(pose[1])[:3, :3]
            t = np.array(pose[0])
            cld_map = cld_cam @ R.T + t  # (h, w, 3)
            clds.append(cld_map)
        # visualize the point cloud
        print("sampling done, visualizing the point cloud")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.reshape(clds, (-1, 3)))
        o3d.visualization.draw_geometries([pcd])
        clds = np.array(clds, dtype=np.float32)
        np.save(model_path, (imgs_rgb, clds))
        print("model saved to ", model_path)

    def snapshot(self):
        """take a snapshot of the current scene"""
        while not self.tf_buf.can_transform(
            self.coord_map, self.coord_cam, rospy.Time(0)
        ):
            rospy.sleep(0.1)
        trans = self.tf_buf.lookup_transform(
            self.coord_map, self.coord_cam, rospy.Time(0)
        )
        pose = [
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
            ],
            [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ],
        ]
        with self.lock_rgb:
            img_rgb = deepcopy(self.img_rgb)
        with self.lock_depth:
            img_dep = deepcopy(self.img_dep)
        return img_rgb, img_dep, pose

    def on_rec_img(self, msg):
        with self.lock_rgb:
            self.img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def on_rec_depth(self, msg):
        with self.lock_depth:
            self.img_dep = (
                self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1") / 1000.0
            )


if __name__ == "__main__":
    ml = ModelLib(load_only=False)
    imgs_rgb, clds = ml.load("carbinet")
    ml.visualize_snapshots(imgs_rgb, clds)
