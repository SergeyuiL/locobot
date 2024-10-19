# !/usr/bin/env python3
import numpy as np
from copy import deepcopy
import os
import cv2
from threading import Thread

import tf, tf2_ros
import rospy
from cv_bridge import CvBridge

from locobot.srv import setgoal, setgoalRequest, setgoalResponse
from perception_service.srv import Sam2Gpt4Infer, Sam2Gpt4InferRequest, Sam2Gpt4InferResponse
from perception_service.srv import GraspInfer, GraspInferRequest, GraspInferResponse

from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped, Point, Vector3, PoseArray, Quaternion
import tf2_geometry_msgs


def Point2Vector(p: Point):
    v = Vector3() 
    for key in ['x', 'y', 'z']:
        setattr(v, key, getattr(p, key))
    return v

def Vector2Point(v: Vector3):
    p = Point()
    for key in ['x', 'y', 'z']:
        setattr(p, key, getattr(v, key))
    return p

def reconstruct(depth:np.ndarray, cam_intrin:np.ndarray, rgb=None):
    """ reconstruct point cloud from depth image """
    if len(depth) < 2:
        return None
    w, h = depth.shape[-2:]
    if w == 0 or h == 0:
        return None
    u, v = np.meshgrid(np.arange(h), np.arange(w), indexing='xy')
    z = depth
    x = (u - cam_intrin[0, 2]) * z / cam_intrin[0, 0]
    y = (v - cam_intrin[1, 2]) * z / cam_intrin[1, 1]
    cld = np.stack([x, y, z], axis=-1)
    rgb_cld = np.concatenate([cld, rgb], axis=-1) if rgb is not None else cld
    return cld, rgb_cld

class FBI:
    def __init__(self):
        rospy.init_node('FBI', anonymous=True)
        print("init node in " + __file__)

        rospy.wait_for_service('/locobot/arm_control')
        print("all_control.py is up")
        cam_info:CameraInfo = rospy.wait_for_message('/locobot/camera/color/camera_info', CameraInfo)
        print("RGBD camera is up")
        rospy.wait_for_service('/grasp_infer')
        print("GraspNet is up")
        rospy.wait_for_service('/sam2gpt4_infer')
        print("sam2gpt4 is up")

        ## actuation API
        self.arm_ctl = rospy.ServiceProxy('/locobot/arm_control', setgoal)
        self.chassis_ctl = rospy.ServiceProxy('/locobot/chassis_control', setgoal)
        self.gripper_ctl = rospy.ServiceProxy('/locobot/gripper_control', SetBool)
        self.cam_yaw_ctl = rospy.ServiceProxy('/locobot/camera_yaw_control', SetBool)
        self.cam_pitch_ctl = rospy.ServiceProxy('/locobot/camera_pitch_control', SetBool)

        ## perception API
        self.grasp_det = rospy.ServiceProxy('/grasp_infer', GraspInfer)
        self.seg_det = rospy.ServiceProxy('/sam2gpt4_infer', Sam2Gpt4Infer)

        ## params
        self.cam_intrin = np.array(cam_info.K).reshape((3, 3))
        self.coord_map = "map"
        self.coord_arm_base = "locobot/base_footprint"
        # self.coord_grasp = "locobot/grasp_goal"
        self.coord_cam = "locobot/camera_depth_link" ## same with "locobot/camera_color_frame"
        ## TODO: look up the coordinate of gripper in rviz
        # self.coord_gripper = "locobot/"

        ## caller
        self.bridge = CvBridge()
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)
        self.tf_pub = tf2_ros.TransformBroadcaster()

        ## subscribe image
        rospy.Subscriber('/locobot/camera/color/image_raw', Image, self.on_rec_img)
        rospy.Subscriber('/locobot/camera/aligned_depth_to_color/image_raw', Image, self.on_rec_depth)
        self.pub_cld = rospy.Publisher('/locobot/point_cloud', PointCloud2, queue_size=1)

        rospy.wait_for_message('/locobot/camera/color/image_raw', CameraInfo)
        rospy.wait_for_message('/locobot/camera/aligned_depth_to_color/image_raw', CameraInfo)

        mask = self.get_mask("handle")
        cv2.imwrite("mask.png", mask.astype(np.uint8)*255)
        print("mask got")
        self.grasp(mask)
    
    def grasp(self, mask):
        """ grasp handle """
        rgb = deepcopy(self.img_rgb)
        depth = deepcopy(self.img_dep)
        cld = deepcopy(self.cld)

        print("sending grasp infer request")
        msg = GraspInferRequest(
            mask = self.bridge.cv2_to_imgmsg(mask, encoding="mono8"),
            cloud = self.create_pc2_msg(cld, rgb)
        )
        
        print("generating grasps from GraspNet service")
        res:GraspInferResponse = self.grasp_det(msg)
        grasps = res.pose
        if not res.result or not len(grasps):
            return
        print("grasps generated")
        
        ## `goal` is in the same link with `depth`
        goal = self.filt_grps(grasps)
        Thread(target=self.pub_grp, args=(goal,)).start()
        if (input(f"goal is {goal}. press enter to continue") == "q"):
            return
        self.arm_ctl(goal)

        print("trying to grasp")
        self.gripper_ctl(SetBoolRequest(data=True))
        print("finish trying")
        return 

    def approach(self, goal:Pose):
        """ approach to the grasp goal """
        ## navigate to the front of grasp
        trans = self.tf_buf.lookup_transform(self.coord_grasp, self.coord_map, rospy.Time(0))
        p = Pose()
        p.position.x = -0.2 ## offset
        p.orientation.w = 1.0
        navi_goal = tf2_geometry_msgs.do_transform_pose(p, trans)
        ## call navigation service
        self.chassis_ctl(navi_goal.pose)
        return

    def filt_grps(self, grasps:PoseArray):
        ## TODO: get best grasp as goal
        goal = grasps[0]
        return goal
    
    def pub_grp(self, goal:Pose):
        """ publish transform of grasp goal to tf """
        pub = rospy.Publisher('/locobot/grasp_goal', PoseStamped, queue_size=1)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.coord_cam
            msg.pose = goal
            pub.publish(msg)
            rate.sleep()
        return 

    def get_mask(self, prompt:str):
        """ call segmentation service to get mask of `prompt` """
        rgb = deepcopy(self.img_rgb)
        res:Sam2Gpt4InferResponse = self.seg_det(prompt, self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8"))
        if not res.result:
            return None
        return self.bridge.imgmsg_to_cv2(res.mask, desired_encoding="mono8")

    def on_rec_img(self, msg:Image):
        self.img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        return

    def on_rec_depth(self, msg:Image):
        self.img_dep = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1") / 1000.0
        self.cld, _ = reconstruct(self.img_dep, self.cam_intrin)
        hd = Header(frame_id=self.coord_cam, stamp=rospy.Time.now())
        fds = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        msg_pc2 = pc2.create_cloud(hd, fds, np.reshape(self.cld, (-1, 3)))
        self.pub_cld.publish(msg_pc2)
        return
    
    def create_pc2_msg(self, cld, rgb):
        ## cloud_rgb
        hd = Header(frame_id=self.coord_cam, stamp=rospy.Time.now())
        fds = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1),
        ]
        msg_pc2 = pc2.create_cloud(hd, fds, np.reshape(np.concatenate([cld, rgb], axis=-1), (-1, 6)))
        return msg_pc2

    

if __name__ == '__main__':
    fbi = FBI()
    rospy.spin()