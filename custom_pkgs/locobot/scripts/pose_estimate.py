#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import os
import cv2
import threading
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import sys

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


def transform_cld(cld: np.ndarray, pose: np.ndarray):
    """transform the point cloud with pose"""
    R = tf.transformations.quaternion_matrix(pose[1])[:3, :3]
    t = np.array(pose[0])
    cld_map = cld @ R.T + t
    return cld_map


class ModelLib:
    def __init__(self, load_only):
        """load_only: if True, only load the model library without ROS dependencies. if `sample_wizard()` is needed, set it to False"""
        self.model_folder = os.path.join(os.path.dirname(__file__), "ModelLib")
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        self.load_only = load_only
        if not load_only:
            rospy.init_node("pose_estimate", anonymous=True)
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
        model_path = os.path.join(self.model_folder, model_name + ".pt")
        imgs_rgb, clds, poses = pickle.load(open(model_path, 'rb'))
        imgs_rgb = np.asarray(imgs_rgb, dtype=np.uint8)
        clds = np.asarray(clds, dtype=np.float32)
        return imgs_rgb, clds, poses

    def visualize_snapshots(self, imgs_rgb, clds, poses=None):
        """visualize the point cloud of the model"""
        print("visualizing the point cloud")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.reshape([transform_cld(cld, poses[i]) for i, cld in enumerate(clds)], (-1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.reshape(imgs_rgb, (-1, 3)) / 255.0)
        o3d.visualization.draw_geometries([pcd])
        # visualize the point cloud with color

    def sample_wizard(self, model_name):
        """sample a few snapshots of the model with given name"""
        model_path = os.path.join(self.model_folder, model_name + ".pt")
        print("Press 's' to save the current snapshot")
        print("Press 'q' to quit")
        snapshots = []  # list of (rgb, depth, pose)
        while True:
            if hasattr(self, "img_rgb"):
                bgr = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("camera", bgr)
                key = cv2.waitKey(33)
            else:
                print("waiting for camera image")
                rospy.sleep(0.1)
                continue
            if key == ord("s"):
                print("saving snapshot")
                snapshots.append(self.snapshot())
            elif key == ord("q"):
                print("quitting")
                break
        imgs, clds, poses = [], [], []
        for rgb, dep, pose in snapshots:
            imgs.append(rgb)
            poses.append(pose)
            cld_cam = reconstruct(dep, self.cam_intrin)  # (h, w, 3)
            clds.append(cld_cam)
        # visualize the point cloud
        print("sampling done, visualizing the point cloud")
        self.visualize_snapshots(imgs, clds, poses)
        # save the model snapshots
        clds = np.asarray(clds, dtype=np.float32)  # (N, H, W, 3)
        imgs = np.asarray(imgs, dtype=np.uint8)  # (N, H, W, 3)
        pickle.dump((imgs, clds, poses), open(model_path, "wb"))
        print("model saved to ", model_path)
        np.set_printoptions(precision=3, suppress=True)
        for i, pose in enumerate(poses):
            pos = np.array(pose[0])
            quat = np.array(pose[1])
            print(f"pose {i}: {pos}, {quat}")

    def snapshot(self):
        """take a snapshot of the current scene"""
        while not self.tf_buf.can_transform(
            self.coord_map, self.coord_cam, rospy.Time(0)
        ):
            print("waiting for tf")
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


from lightglue import LightGlue, SuperPoint, DISK, SIFT, DoGHardNet, ALIKED
from lightglue.utils import load_image, rbd, numpy_image_to_torch, match_pair
from lightglue import viz2d
import torch

torch.set_grad_enabled(False)


class PoseEstimator:
    @staticmethod
    def match_SP_LG(rgb_src, cld_src, rgbs_dst: np.ndarray, clds_dst):
        """Match rgb_src and rgb_dst with SuperPoint extractor and LightGlue matcher
        rgb_src: (H1, W1, 3)
        cld_src: (H1, W1, 3)
        rgbs_dst: (N, H2, W2, 3)
        clds_dst: (N, H2, W2, 3)
        """
        if not hasattr(PoseEstimator, "superpoint"):
            PoseEstimator.superpoint = SuperPoint(max_num_keypoints=1024).eval()
        if not hasattr(PoseEstimator, "lightglue"):
            PoseEstimator.lightglue = LightGlue(features="superpoint").eval()
        assert len(rgbs_dst.shape) == 4
        N = rgbs_dst.shape[0]
        image0 = numpy_image_to_torch(rgb_src)
        feats0 = PoseEstimator.superpoint.extract(image0)
        pixs_src, pixs_dst = [], []
        for i in range(N):
            print(f"matching No. {i} image")
            image1 = numpy_image_to_torch(rgbs_dst[i])
            feats1 = PoseEstimator.superpoint.extract(image1)
            matches01 = PoseEstimator.lightglue({"image0": feats0, "image1": feats1})
            feats1, matches01 = rbd(feats1), rbd(matches01)
            kpts0, kpts1, matches = (
                rbd(feats0)["keypoints"],
                feats1["keypoints"],
                matches01["matches"],
            )
            m_kpts0, m_kpts1 = (
                kpts0[matches[..., 0]],
                kpts1[matches[..., 1]],
            )

            axes = viz2d.plot_images([image0, image1])
            viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
            viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')

            kpc0 = viz2d.cm_prune(matches01["prune0"])
            kpc1 = viz2d.cm_prune(matches01["prune1"])
            viz2d.plot_images([image0, image1])
            viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
            plt.show()

            pixs_src.append(m_kpts0)
            pixs_dst.append(m_kpts1)
        return pixs_src, pixs_dst

    @staticmethod
    def match_SIFT_FLANN(rgb_src, cld_src, rgbs_dst: np.ndarray, clds_dst):
        """Match rgb_src and rgb_dst with SIFT extractor and FLANN matcher
            rgb_src: (H1, W1, 3)
            cld_src: (H1, W1, 3)
            rgbs_dst: (N, H2, W2, 3)
            clds_dst: (N, H2, W2, 3)
        return:
            pixs_src: (N, Ln, 2)
            pixs_dst: (N, Ln, 2)
        """
        if not hasattr(PoseEstimator, "sift"):
            PoseEstimator.sift = cv2.SIFT_create()
        if not hasattr(PoseEstimator, "flann"):
            PoseEstimator.flann = cv2.FlannBasedMatcher()
        kp1, des1 = PoseEstimator.sift.detectAndCompute(rgb_src, None)
        pixs_src, pixs_dst = [], []
        for i in range(len(rgbs_dst)):
            print(f"matching model image No.{i}")
            kp2, des2 = PoseEstimator.sift.detectAndCompute(rgbs_dst[i], None)
            if len(kp2) == 0:
                print(f"no feature detected in model image No.{i}")
                continue
            res = PoseEstimator.flann.knnMatch(des1, des2, k=2)
            # Lowe distance ratio test, default threshold is 0.7
            matches = [m for m, n in res if m.distance < 0.7 * n.distance]
            pixs1, pixs2 = PoseEstimator.get_match_pixs(kp1, kp2, matches)
            mask = PoseEstimator.filter_matches(
                pixs1, pixs2, cld_src, clds_dst[i], rgb_src, rgbs_dst[i]
            )
            pixs1, pixs2 = pixs1[mask], pixs1[mask]
            # visualization
            img3 = cv2.drawMatches(
                rgb_src,
                kp1,
                rgbs_dst[i],
                kp2,
                matches,
                None,
                matchesMask=np.array(mask, dtype=int),
                flags=2,
            )
            cv2.imshow("match", img3)
            cv2.waitKey(0)
            pixs_src.append(pixs1)
            pixs_dst.append(pixs2)
        return pixs_src, pixs_dst

    @staticmethod
    def estimate():
        pass

    @staticmethod
    def filter_matches(pixs1, pixs2, cld1, cld2, rgb1, rgb2):
        """
        pixs1, pixs2: matched image points, shaped like (L, 2)
        return 0-1 mask of the same length with pixs1/pixs2
        """
        L = len(pixs1)
        # other filter
        # ...

        # homography mask
        # homoMask: 0-1 mask
        if L < 4:
            homoMask = np.ones((L,), dtype=int)
        else:
            M, homoMask = cv2.findHomography(pixs1, pixs2, cv2.RANSAC)
            homoMask = homoMask.ravel()  # (L, )

        # pick out points with depth available
        # depthMask: 0-1 mask
        pts1 = cld1[pixs1[:, 1], pixs1[:, 0]]
        pts2 = cld2[pixs2[:, 1], pixs2[:, 0]]
        depthMask = (np.linalg.norm(pts1, axis=1) > 1e-3) & (
            np.linalg.norm(pts2, axis=1) > 1e-3
        )
        print(f"filt out {sum(1-depthMask)}")
        mask = homoMask  # & depthMask
        return np.array(mask, dtype=bool)

    @staticmethod
    def get_match_pixs(kp1, kp2, matches):
        """extract matched 2D points in `matches`"""
        pixs1 = np.int32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pixs2 = np.int32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return pixs1, pixs2


if __name__ == "__main__":
    """ visualize the point cloud of the model """
    # ml = ModelLib(load_only=True)
    # rgbs_dst, clds_dst, poses = ml.load("carbinet")
    # ml.visualize_snapshots(rgbs_dst, clds_dst, poses)

    """ sample """
    ml = ModelLib(load_only=False)
    if len(sys.argv) < 2:
        print("please specify the model name")
        exit(0)
    model_name = sys.argv[1]
    ml.sample_wizard(model_name)