#!/usr/bin/env python3
import os, cv2, json
import argparse
import numpy as np
from copy import deepcopy
from threading import Lock

import tf, tf2_ros
import tf.transformations
import rospy
from cv_bridge import CvBridge

from std_srvs.srv import SetBool
from std_msgs.msg import Header

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import *
from moveit_msgs.msg import OrientationConstraint, Constraints

import tf2_geometry_msgs

from ultralytics import YOLO

# custom srv
from locobot.srv import SetPose2D, SetFloat32
from perception_service.srv import GraspInfer, GraspInferRequest, GraspInferResponse
from perception_service.srv import GroundedSam2Infer, GroundedSam2InferRequest, GroundedSam2InferResponse

# custom python module
from arm_control import LocobotArm
from wbc import LocobotMPC


def Point2Vector(p: Point):
    v = Vector3()
    for key in ["x", "y", "z"]:
        setattr(v, key, getattr(p, key))
    return v


def Vector2Point(v: Vector3):
    p = Point()
    for key in ["x", "y", "z"]:
        setattr(p, key, getattr(v, key))
    return p


def transform_pose(pose: Pose, dst_frame: str, src_frame: str, tf_buf: tf2_ros.Buffer) -> Pose:
    """transform pose from `src_frame` to `dst_frame` now"""
    while not tf_buf.can_transform(dst_frame, src_frame, rospy.Time(0)):
        rospy.sleep(0.1)
    msg = PoseStamped()
    msg.pose = pose
    msg.header.frame_id = src_frame
    msg.header.stamp = rospy.Time(0)
    return tf_buf.transform(msg, dst_frame).pose


def transform_vec(vec: Vector3, dst_frame: str, src_frame: str, tf_buf: tf2_ros.Buffer, trans=None) -> Vector3:
    """transform vector from `src_frame` to `dst_frame` now. use `trans` if it's available"""
    if trans is None:
        while not tf_buf.can_transform(dst_frame, src_frame, rospy.Time(0)):
            rospy.sleep(0.2)
        trans_stamped = tf_buf.lookup_transform(dst_frame, src_frame, rospy.Time(0))
    else:
        trans_stamped = TransformStamped()
        trans_stamped.header.stamp = rospy.Time(0)
        trans_stamped.transform = trans
        trans_stamped.header.frame_id = dst_frame
        trans_stamped.child_frame_id = src_frame
    msg = Vector3Stamped()
    msg.vector = vec
    msg.header.frame_id = src_frame
    msg.header.stamp = rospy.Time(0)
    return tf2_geometry_msgs.do_transform_vector3(msg, trans_stamped).vector


def translate(pose: Pose, vec: Vector3) -> Pose:
    """translate pose by vector (assume they are in same coordinate, if not, use @transform_vec first)"""
    res = deepcopy(pose)
    res.position.x += vec.x
    res.position.y += vec.y
    res.position.z += vec.z
    return res


def reconstruct(depth: np.ndarray, cam_intrin: np.ndarray, rgb=None):
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
    rgb_cld = np.concatenate([cld, rgb], axis=-1) if rgb is not None else cld
    return cld, rgb_cld


class Demo:
    def __init__(self):
        rospy.init_node("demo", anonymous=True)
        print("---------- init Demo ----------")
        self.CHAS_PT1 = Point(x=0.15, y=-0.6, z=0)
        self.CHAS_PT2 = Point(x=0.15, y=0, z=0)
        self.ARM_PT1 = Point(x=0.35, y=0, z=0.5)
        self.ARM_PT2 = Point(x=0.4, y=0, z=0.4)
        self.init_vars()
        self.init_caller()
        self.wait_services()
        # publish grasp goal (if possible)
        rospy.Timer(rospy.Duration(0.1), self.pub_grp)
        print("---------- init done ----------")

    def init_vars(self):
        cam_info: CameraInfo = rospy.wait_for_message("/locobot/camera/color/camera_info", CameraInfo)
        # params
        self.t0 = rospy.Time.now()
        self.map = {}
        self.goal = None  # goal pose in map frame
        self.img_rgb = None
        self.img_dep = None
        self.lock_rgb = Lock()
        self.lock_dep = Lock()
        # constant
        self.cam_intrin = np.array(cam_info.K).reshape((3, 3))
        self.coord_map = "map"
        self.coord_arm_base = "locobot/base_footprint"
        self.coord_gripper = "locobot/ee_gripper_link"
        self.coord_ee_goal = "locobot/ee_goal"
        self.coord_cam = "locobot/camera_color_optical_frame"  # not locobot/camera_aligned_depth_to_color_frame
        self.coord_targ = "pose_estimate/target"

        # ros objects
        self.bridge = CvBridge()
        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        self.tf_spub = tf2_ros.StaticTransformBroadcaster()
        # YOLO object
        # self.model = YOLO()

    def init_caller(self):
        # actuation API
        self.arm = LocobotArm(serving=False)
        self.chassis_ctl = rospy.ServiceProxy("/locobot/chassis_control", SetPose2D)
        self.gripper_ctl = rospy.ServiceProxy("/locobot/gripper_control", SetBool)
        self.cam_yaw_ctl = rospy.ServiceProxy("/locobot/camera_yaw_control", SetFloat32)
        self.cam_pitch_ctl = rospy.ServiceProxy("/locobot/camera_pitch_control", SetFloat32)
        self.chassis_vel = rospy.Publisher("/locobot/mobile_base/commands/velocity", Twist, queue_size=10)
        # perception API
        self.grasp_det = rospy.ServiceProxy("/grasp_infer", GraspInfer)
        self.seg_det = rospy.ServiceProxy("/grounded_sam2_infer", GroundedSam2Infer)
        # visualization
        self.pub_cld = rospy.Publisher("/locobot/point_cloud", PointCloud2, queue_size=1)  # for visualization and debug

    def wait_services(self):
        rospy.Subscriber("/locobot/camera/color/image_raw", Image, self.on_rec_img)
        rospy.Subscriber("/locobot/camera/aligned_depth_to_color/image_raw", Image, self.on_rec_depth)

        rospy.wait_for_service("/locobot/arm_control")
        print("all_control.py is up")
        rospy.wait_for_message("/locobot/camera/color/image_raw", Image)
        rospy.wait_for_message("/locobot/camera/aligned_depth_to_color/image_raw", Image)
        print("RGBD camera is up")
        # rospy.wait_for_service("/grasp_infer")
        # print("GraspNet is up")
        # rospy.wait_for_service("/grounded_sam2_infer")
        # print("grounded_sam is up")

    def main_dev(self):
        with self.lock_rgb:
            rgb = deepcopy(self.img_rgb)
        with self.lock_dep:
            cld = deepcopy(self.cld)

        _ = self.model.predict(rgb)

        # TODO: a list of [label, mask]
        ...
        results = []
        ...

        for label, mask in results:
            pts = cld[np.nonzero(mask)]  # (n, 3)
            pts = pts[np.nonzero(pts[2] > 0.1), :]  # filter out points with depth < 0.1
            pos = pts.mean()
            t = transform_vec(Vector3(x=pos[0], y=pos[1], z=pos[2]), self.coord_map, self.coord_cam, self.tf_buf)
            pos_glb = np.array([t.x, t.y, t.z])
            if label not in self.map:
                print(f"add {label} to map at {pos_glb} at {rospy.Time.now() - self.t0:.1f} sec")
                self.map[label] = pos_glb
            else:
                self.map[label] = pos_glb * 0.1 + self.map[label] * 0.9  # low-pass

    def grasp_mask(self, rgb, cld, mask):
        """grasp given object defined by `mask`"""
        msg = GraspInferRequest(
            mask=self.bridge.cv2_to_imgmsg(mask, encoding="mono8"), cloud=self.create_pc2_msg(cld, rgb)
        )
        t0 = rospy.Time.now()
        resp: GraspInferResponse = self.grasp_det(msg)
        grasps = resp.pose
        if not resp.result or not len(grasps):
            return
        print(f"grasps generated ({(rospy.Time.now() - t0).to_sec():.1f} sec)")
        # `goal` is in the same link with `depth`
        goal = self.filt_grps(grasps)
        goal = transform_pose(goal, self.coord_map, self.coord_cam, self.tf_buf)
        self.grasp_goal(goal)

    def reach_goal_with_direction(self, goal_pose, offset):
        """reach `goal_pose + offset` first and then `goal_pose`.
        Note: `goal_pose` is w.r.t. the map frame.
        """
        self.goal = goal_pose
        self.pub_grp(None)
        rospy.sleep(0.1)

        # reach pre-goal (goal_pose + offset)
        vec = Vector3(*offset)  # offset is list-like, e.g., [-0.1, 0, 0]
        t = transform_vec(vec, self.coord_map, self.coord_ee_goal, self.tf_buf)
        goal_pre = translate(goal_pose, t)
        ee_goal = transform_pose(goal_pre, self.coord_arm_base, self.coord_map, self.tf_buf)
        self.arm.move_to_poses([ee_goal])

        # add orientation constraint
        oc = OrientationConstraint()
        oc.header.frame_id = self.coord_arm_base
        oc.link_name = self.arm.arm_group.get_end_effector_link()
        oc.orientation = ee_goal.orientation  # 约束的目标朝向
        oc.absolute_x_axis_tolerance = 0.15  # -0.15 ~ +0.15 rad
        oc.absolute_y_axis_tolerance = 0.15
        oc.absolute_z_axis_tolerance = 0.15
        oc.weight = 1.0
        constraints = Constraints()
        constraints.orientation_constraints.append(oc)
        self.arm.arm_group.set_path_constraints(constraints)

        # reach goal_pose
        ee_goal = transform_pose(goal_pose, self.coord_arm_base, self.coord_map, self.tf_buf)
        self.arm.move_to_poses([ee_goal])

        # clear constraint
        self.arm.arm_group.clear_path_constraints()
        return

    def grasp_goal(self, goal_pose):
        """aprroach from negative x-axis and then grasp"""
        self.gripper_ctl(False)
        self.reach_goal_with_direction(goal_pose, offset=[-0.08, 0, 0])
        self.gripper_ctl(True)
        return

    def place_goal(self, goal_pose):
        """approach from positive z-axis and then place"""
        self.reach_goal_with_direction(goal_pose, offset=[0, 0, 0.08])
        self.gripper_ctl(False)
        return

    def authenticate(self, description: str = ""):
        """
        authenticate arm executation for goal pose (ask for 'enter' in terminal)
        user can check the goal pose ("locobot/ee_goal", "TransformStamped") in rviz
        """
        if input(f"<Confirm {description} with Enter:>\n") != "":
            print("aborted")
            exit(1)
        return

    def filt_grps(self, grasps: PoseArray):
        # TODO: get best grasp as goal
        goal = grasps[0]
        return goal

    def pub_grp(self, timer_event):
        """publish transform of grasp goal to tf"""
        if self.goal is None or rospy.is_shutdown():
            return

        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.coord_map
        msg.child_frame_id = self.coord_ee_goal
        msg.transform.translation = Point2Vector(self.goal.position)
        msg.transform.rotation = self.goal.orientation
        self.tf_pub.sendTransform(msg)

    def get_mask(self, rgb, prompt: str):
        """call segmentation service to get mask of `prompt`"""
        cv2.imwrite(
            os.path.join(os.path.dirname(__file__), f"mask/{prompt}_input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        resp: GroundedSam2InferResponse = self.seg_det(prompt, self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8"))
        if not resp.result:
            return None
        for i in range(len(resp.masks)):
            mask = self.bridge.imgmsg_to_cv2(resp.masks[i], desired_encoding="mono8")
            cv2.imwrite(os.path.join(os.path.dirname(__file__), f"mask/{prompt}_{i}.png"), mask.astype(np.uint8))
        mask = self.filt_masks(prompt, resp)
        if mask is not None:
            cv2.imwrite(os.path.join(os.path.dirname(__file__), f"mask/{prompt}_output.png"), mask.astype(np.uint8))
        return mask

    def filt_masks(self, prompt, resp: GroundedSam2InferResponse):
        """filter with rules"""
        masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in resp.masks]
        if "handle" not in prompt:
            return masks[0]

        boxes = np.array(resp.bounding_boxes, dtype=int).reshape((-1, 4)).tolist()
        print(boxes)
        box = min([box for box in boxes if box[1] >= 20], key=lambda box: box[1])
        idx = boxes.index(box)
        return masks[idx]

    def on_rec_img(self, msg: Image):
        with self.lock_rgb:
            self.img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def on_rec_depth(self, msg: Image):
        with self.lock_dep:
            self.img_dep = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1") / 1000.0
            self.cld, _ = reconstruct(self.img_dep, self.cam_intrin)

    def create_pc2_msg(self, cld, rgb):
        # cloud_rgb
        hd = Header(frame_id=self.coord_cam, stamp=rospy.Time.now())
        fds = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("r", 12, PointField.FLOAT32, 1),
            PointField("g", 16, PointField.FLOAT32, 1),
            PointField("b", 20, PointField.FLOAT32, 1),
        ]
        msg_pc2 = pc2.create_cloud(hd, fds, np.reshape(np.concatenate([cld, rgb], axis=-1), (-1, 6)))
        return msg_pc2

    def demo_goal(self, path_save, object_label: str):
        print("Start demo goal wizard.")

        pose_targ2cam = transform_pose(
            Pose(position=Point(), orientation=Quaternion(0, 0, 0, 1)), self.coord_cam, self.coord_targ, self.tf_buf
        )

        print("Now, manually set the robot arm to the goal pose.")
        print("Do NOT move camera during that process.")
        print("When you are done, press <Enter> to confirm, otherwise to abort> ", end="")

        if input() != "":
            print("demo grasp aborted.")
            return

        # get goal pose (in coord_targ)
        pose_targ2goal = transform_pose(pose_targ2cam, self.coord_gripper, self.coord_cam, self.tf_buf)

        pos = pose_targ2goal.position
        rot = pose_targ2goal.orientation
        euler = tf.transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
        M = tf.transformations.compose_matrix(translate=[pos.x, pos.y, pos.z], angles=euler)
        M_inv = tf.transformations.inverse_matrix(M)
        _, _, angles, pos, _ = tf.transformations.decompose_matrix(M_inv)
        rot = tf.transformations.quaternion_from_euler(*angles)
        # save to output_file
        with open(path_save, "r") as f:
            content = json.load(f)
        content[object_label] = {"pos": list(pos), "rot": list(rot)}
        with open(path_save, "w") as f:
            json.dump(content, f, indent=4)
        print(f"write goal pose w.r.t. '{self.coord_targ}' to {path_save}")
        return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control Locobot to grasp/place objects.")
    parser.add_argument(
        "--label", type=str, required=True, help="The label for target object, e.g. handle, rod, wire, etc."
    )
    ex_group = parser.add_mutually_exclusive_group()
    ex_group.add_argument("--grasp", action="store_true")
    ex_group.add_argument("--place", action="store_true")
    args = parser.parse_args()

    object_label = args.label

    # reset_arm()
    demo = Demo()

    path_gpose = "./goal_pose.json"  # path to goal_pose.json
    if not os.path.exists(path_gpose):
        json.dump({}, path_gpose)

    with open(path_gpose, "r") as f:
        content = json.load(f)

    if object_label not in content:
        content = demo.demo_goal(path_gpose, object_label)

    p = content[object_label]["pos"]
    q = content[object_label]["rot"]

    # publish static transformation
    msg = TransformStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = demo.coord_targ
    msg.child_frame_id = demo.coord_ee_goal
    msg.transform.translation = Vector3(*p)
    msg.transform.rotation = Quaternion(*q)
    demo.tf_spub.sendTransform(msg)

    demo.authenticate("Perform grasp/place.")

    goal = transform_pose(
        Pose(position=Point(0, 0, 0), orientation=Quaternion(0, 0, 0, 1)),
        demo.coord_map,
        demo.coord_ee_goal,
        demo.tf_buf,
    )
    if args.grasp:
        demo.grasp_goal(goal)
    else:
        demo.place_goal(goal)

    print("done")
    rospy.spin()
