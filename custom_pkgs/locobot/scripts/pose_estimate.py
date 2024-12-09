#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import os
import cv2
import threading
from copy import deepcopy
import matplotlib.pyplot as plt

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
        model_path = os.path.join(self.model_folder, model_name + ".npy")
        imgs_rgb, clds = np.load(model_path)
        imgs_rgb = imgs_rgb.astype(np.uint8)
        clds = clds.astype(np.float32)
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
                print("waiting for camera image")
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
        # save the model snapshots
        clds = np.array(clds, dtype=np.float32)
        imgs_rgb = np.array(imgs_rgb, dtype=np.uint8)
        np.save(model_path, (imgs_rgb, clds))
        print("model saved to ", model_path)

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
    extractor = SuperPoint(max_num_keypoints=1024).eval()
    matcher = LightGlue(features="superpoint").eval()

    @staticmethod
    def match(rgb_src, cld_src, rgbs_dst: np.ndarray, clds_dst):
        """Match rgb_src and rgb_dst, and solve pose from 3d correspondance
        rgb_src: (H1, W1, 3)
        cld_src: (H1, W1, 3)
        rgbs_dst: (N, H2, W2, 3)
        clds_dst: (N, H2, W2, 3)
        """
        assert len(rgbs_dst.shape) == 4
        N = rgbs_dst.shape[0]
        image0 = numpy_image_to_torch(rgb_src)
        feats0 = PoseEstimator.extractor.extract(image0)
        for i in range(N):
            print(f"matching No. {i} image")
            image1 = numpy_image_to_torch(rgbs_dst[i])
            feats1 = PoseEstimator.extractor.extract(image1)
            matches01 = PoseEstimator.matcher({"image0": feats0, "image1": feats1})
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


if __name__ == "__main__":
    ml = ModelLib(load_only=False)
    cam_in = ml.cam_intrin
    imgs_rgb, clds = ml.load("carbinet")
    # ml.visualize_snapshots(imgs_rgb, clds)
    rgb, depth, pose = ml.snapshot()
    cld = reconstruct(depth, cam_in)
    R = tf.transformations.quaternion_matrix(pose[1])[:3, :3]
    t = np.array(pose[0])
    cld = cld @ R.T + t  # (h, w, 3)

    PoseEstimator.match(rgb, cld, imgs_rgb, clds)

    # feature extraction
    detector = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher()
    kp1, des1 = detector.detectAndCompute(rgb, None)
    src_pts = []
    dst_pts = []
    src_pixs = []
    for i, img in enumerate(imgs_rgb):
        print(f"matching model image No.{i}")
        kp2, des2 = detector.detectAndCompute(img, None)
        if len(kp2) == 0:
            print(f"no feature detected in model image No.{i}")
            continue
        res = flann.knnMatch(des1, des2, k=2)
        # Lowe distance ratio test, default threshold is 0.7
        matches = []
        for m, n in res:
            if m.distance < 0.7 * n.distance:
                matches.append(m)

        pixs1 = np.int32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pixs2 = np.int32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        cld1 = cld[pixs1[:, 1], pixs1[:, 0]]
        cld2 = clds[i][pixs2[:, 1], pixs2[:, 0]]
        depth_avail = (np.linalg.norm(cld1, axis=1) > 1e-3) & (
            np.linalg.norm(cld2, axis=1) > 1e-3
        )
        print(f"filt out {sum(1-depth_avail)}")
        matches = [matches[i] for i in range(len(matches)) if depth_avail[i]]

        # visualize the matches
        vis = cv2.drawMatches(rgb, kp1, img, kp2, matches[:20], None, flags=2)
        cv2.imshow("matches", vis)
        cv2.waitKey(0)

        src_pixs.append(pixs1[depth_avail])
        src_pts.append(cld1[depth_avail])
        dst_pts.append(cld2[depth_avail])

    src_pixs = np.concatenate(src_pixs, axis=0)  # (L, 2)
    src_pts = np.concatenate(src_pts, axis=0)  # (L, 3)
    dst_pts = np.concatenate(dst_pts, axis=0)  # (L, 3)
    # registration with o3d
    idx = np.arange(src_pts.shape[0], dtype=np.int32)
    corres = np.stack([idx, idx], axis=1)  # (L, 2)
    corres = o3d.utility.Vector2iVector(corres)
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts)
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(dst_pts)
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src, dst, corres, 10
    )
    print("----------3d match---------")
    print(result.transformation)

    # transform cld with the result.transformation
    pcd_det = o3d.geometry.PointCloud()
    pcd_det.points = o3d.utility.Vector3dVector(cld.reshape(-1, 3))
    pcd_det.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
    pcd_det.transform(result.transformation)
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(clds[0].reshape(-1, 3))
    pcd_model.colors = o3d.utility.Vector3dVector(imgs_rgb[0].reshape(-1, 3) / 255.0)
    # pcd_model.transform(result.transformation)
    o3d.visualization.draw_geometries([pcd_det, pcd_model])
