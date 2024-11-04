# !/usr/bin/env python3
import numpy as np
from copy import deepcopy
import os
import cv2
from threading import Thread

import tf, tf2_ros
import rospy
from cv_bridge import CvBridge

from locobot.srv import SetPose, SetPoseRequest, SetPoseResponse
from locobot.srv import SetFloat32, SetFloat32Request, SetFloat32Response
from perception_service.srv import Sam2Gpt4Infer, Sam2Gpt4InferRequest, Sam2Gpt4InferResponse
from perception_service.srv import GraspInfer, GraspInferRequest, GraspInferResponse
from perception_service.srv import GroundedSam2Infer, GroundedSam2InferRequest, GroundedSam2InferResponse
from interbotix_xs_msgs.srv import Reboot

from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped, Point, Vector3Stamped, Vector3, PoseArray, Quaternion, Twist
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

def transform_pose(pose:Pose, dst_frame:str, src_frame:str, tf_buf:tf2_ros.Buffer)->Pose:
    """ transform pose from `src_frame` to `dst_frame` now """
    while not tf_buf.can_transform(dst_frame, src_frame, rospy.Time(0)):
        rospy.sleep(0.2)
    msg = PoseStamped()
    msg.pose = pose
    msg.header.frame_id = src_frame
    msg.header.stamp = rospy.Time(0)
    return tf_buf.transform(msg, dst_frame).pose

def transform_vec(vec:Vector3, dst_frame:str, src_frame:str, tf_buf:tf2_ros.Buffer, trans=None)->Vector3:
    """ transform vector from `src_frame` to `dst_frame` now. use `trans` if it's available """
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

def translate(pose:Pose, vec:Vector3)->Pose:
    """ translate pose by vector (assume they are in same coordinate, if not, use @transform_vec first) """
    for idx in ['x', 'y', 'z']:
        setattr(pose.position, idx, getattr(pose.position, idx) + getattr(vec, idx))
    return pose

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

def reset_arm():
    prx = rospy.ServiceProxy('/locobot/reboot_motors', Reboot)
    prx('group', 'arm', True, True)

class FBI:
    def __init__(self):
        rospy.init_node('FBI', anonymous=True)
        print("---------- init FBI ----------")
        self.CHAS_PT1 = Point(x=0.15, y=-0.6, z=0)
        self.CHAS_PT2 = Point(x=0.15, y=0, z=0)
        self.ARM_PT1 = Point(x=0.35, y=0, z=0.5)
        self.ARM_PT2 = Point(x=0.4, y=0, z=0.4)
        self.init_vars()
        self.init_caller()
        self.wait_services()
        ## publish grasp goal (if possible)
        Thread(target=self.pub_grp).start()
        print("---------- init done ----------")
        self.main()
        return
    
    def init_vars(self):
        cam_info:CameraInfo = rospy.wait_for_message('/locobot/camera/color/camera_info', CameraInfo)
        ## params
        self.is_quit = False
        self.goal = None ## goal pose in map frame
        ## constant
        self.cam_intrin = np.array(cam_info.K).reshape((3, 3))
        self.coord_map = "map"
        self.coord_arm_base = "locobot/base_footprint"
        self.coord_grasp = "locobot/grasp_goal"
        self.coord_cam = "locobot/camera_color_optical_frame" ## not locobot/camera_aligned_depth_to_color_frame
        ## ros objects
        self.bridge = CvBridge()
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)
        self.tf_pub = tf2_ros.TransformBroadcaster()
    
    def init_caller(self):
        ## actuation API
        self.arm_ctl = rospy.ServiceProxy('/locobot/arm_control', SetPose)
        self.arm_sleep = rospy.ServiceProxy('/locobot/arm_sleep', SetBool)
        self.chassis_ctl = rospy.ServiceProxy('/locobot/chassis_control', SetPose)
        self.gripper_ctl = rospy.ServiceProxy('/locobot/gripper_control', SetBool)
        self.cam_yaw_ctl = rospy.ServiceProxy('/locobot/camera_yaw_control', SetFloat32)
        self.cam_pitch_ctl = rospy.ServiceProxy('/locobot/camera_pitch_control', SetFloat32)
        self.chassis_vel = rospy.Publisher('/locobot/mobile_base/commands/velocity', Twist, queue_size=10)
        ## perception API
        self.grasp_det = rospy.ServiceProxy('/grasp_infer', GraspInfer)
        self.seg_det = rospy.ServiceProxy('/grounded_sam2_infer', GroundedSam2Infer)
        ## visualization
        self.pub_cld = rospy.Publisher('/locobot/point_cloud', PointCloud2, queue_size=1)   ## for visualization and debug

    def wait_services(self):
        rospy.Subscriber('/locobot/camera/color/image_raw', Image, self.on_rec_img)
        rospy.Subscriber('/locobot/camera/aligned_depth_to_color/image_raw', Image, self.on_rec_depth)

        rospy.wait_for_service('/locobot/arm_control')
        print("all_control.py is up")
        rospy.wait_for_message('/locobot/camera/color/image_raw', Image)
        rospy.wait_for_message('/locobot/camera/aligned_depth_to_color/image_raw', Image)
        print("RGBD camera is up")
        rospy.wait_for_service('/grasp_infer')
        print("GraspNet is up")
        rospy.wait_for_service('/grounded_sam2_infer')
        print("grounded_sam is up")

    def main(self):
        # # navigate to the front of the cabinet
        # self.authenticate("Localization mode")
        # ## TODO: change to detection-based method instead of hard coding here
        # self.chassis_ctl(Pose(position=self.CHAS_PT1, orientation=Quaternion(w=1.0)))
        # self.chassis_ctl(Pose(position=self.CHAS_PT2), orientation=Quaternion(w=1.0)))

        ## step 1: grasp handle
        print("----- grasping handle -----")
        mask = self.get_mask("handle")
        self.grasp(mask)

        print("> pulling handle")
        self.chassis_vel.publish(Twist(linear=Vector3(x=-0.12)))
        rospy.sleep(2)
        self.gripper_ctl(False)

        print("> turning left")
        self.chassis_vel.publish(Twist(angular=Vector3(z=1.2)))
        rospy.sleep(2)
        print("> resetting arm")
        self.arm_sleep(True)

        ## step 2: grasp bowl
        print("----- grasping bowl -----")
        mask = self.get_mask("bowl")
        self.grasp(mask)
        
        print("> holding and rotating")
        self.arm_ctl(Pose(position=self.ARM_PT1, orientation=Quaternion(w=1.0)))
        self.chassis_vel.publish(Twist(angular=Vector3(z=-1.45)))
        rospy.sleep(1.5)
        self.chassis_vel.publish(Twist(linear=Vector3(x=0.1)))
        rospy.sleep(1.5)

        print("> placing obj")
        self.arm_ctl(Pose(position=self.ARM_PT2, orientation=Quaternion(w=1.0)))
        rospy.sleep(3)
        self.gripper_ctl(False)

        print("> resetting arm and grasp handle")
        ## back
        self.chassis_vel.publish(Twist(linear=Vector3(x=-0.14)))
        ## reset arm
        self.arm_sleep(True)
        ## grasp handle again
        print("----- grasping handle -----")
        mask = self.get_mask("handle")
        self.grasp(mask)
        ## push
        self.authenticate("Move")
        self.chassis_vel.publish(Twist(linear=Vector3(x=0.13)))
        ## open gripper
        self.gripper_ctl(False)
        rospy.sleep(2)
        ## back and reset arm
        self.chassis_vel.publish(Twist(linear=Vector3(x=-0.08)))
        self.arm_sleep(True)
        self.quit(0)
        return 
    
    def grasp(self, mask):
        """ grasp given object defined by `mask` """
        rgb = deepcopy(self.img_rgb)
        cld = deepcopy(self.cld)

        # print("sending grasp infer request")
        msg = GraspInferRequest(
            mask = self.bridge.cv2_to_imgmsg(mask, encoding="mono8"),
            cloud = self.create_pc2_msg(cld, rgb)
        )
        # print("generating grasps from GraspNet service")
        resp:GraspInferResponse = self.grasp_det(msg)
        grasps = resp.pose
        if not resp.result or not len(grasps):
            return
        # print("grasps generated")
        ## `goal` is in the same link with `depth`
        goal = self.filt_grps(grasps)
        self.goal = transform_pose(goal, self.coord_map, self.coord_cam, self.tf_buf)
        self.authenticate("Grasp")
        self.gripper_ctl(False)
        ## wait a few seconds for coord_grasp published
        rospy.sleep(0.5)
        ## first move to the front of grasp
        t = transform_vec(Vector3(x=-0.1, y=0, z=0), self.coord_map, self.coord_grasp, self.tf_buf)
        self.goal = translate(self.goal, t) 
        ee_goal = transform_pose(self.goal, self.coord_arm_base, self.coord_map, self.tf_buf)
        self.arm_ctl(ee_goal)
        ## then move to grasp goal
        self.goal = transform_pose(goal, self.coord_map, self.coord_cam, self.tf_buf)
        ee_goal = transform_pose(goal, self.coord_arm_base, self.coord_cam, self.tf_buf)
        resp:SetPoseResponse = self.arm_ctl(ee_goal)
        ## check
        if not resp.result:
            print(f"warning: failed to reach grasp goal, resp.message: {resp.message}")
        #     self.quit(0)
        rospy.sleep(1)
        self.gripper_ctl(True)
        rospy.sleep(2)
        print("finish grasping")
        return
    
    def authenticate(self, description:str=''):
        """
        authenticate arm executation for goal pose (ask for 'enter' in terminal)
        user can check the goal pose ("locobot/grasp_goal", "TransformStamped") in rviz
        """
        if (input(f"<Confirm '{description}' with Enter:>") != ""):
            print("aborted")
            self.quit(1)
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
    
    def pub_grp(self):
        """ publish transform of grasp goal to tf """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.is_quit:
            if self.goal is not None:
                msg = TransformStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = self.coord_map
                msg.child_frame_id = self.coord_grasp
                msg.transform.translation = Point2Vector(self.goal.position)
                msg.transform.rotation = self.goal.orientation
                self.tf_pub.sendTransform(msg)
            rate.sleep()
        return 

    def get_mask(self, prompt:str):
        """ call segmentation service to get mask of `prompt` """
        rgb = deepcopy(self.img_rgb)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), f"mask/{prompt}_input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        resp:GroundedSam2InferResponse = self.seg_det(prompt, self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8"))
        if not resp.result:
            return None
        for i in range(len(resp.masks)):
            mask = self.bridge.imgmsg_to_cv2(resp.masks[i], desired_encoding="mono8")
            cv2.imwrite(os.path.join(os.path.dirname(__file__), f"mask/{prompt}_{i}.png"), mask.astype(np.uint8))
        mask = self.filt_masks(prompt, resp)
        if mask is not None:
            cv2.imwrite(os.path.join(os.path.dirname(__file__), f"mask/{prompt}_output.png"), mask.astype(np.uint8))
        return mask
    
    def filt_masks(self, prompt, resp:GroundedSam2InferResponse):
        """ filter with rules """
        masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding="mono8") for m in resp.masks]
        if 'handle' not in prompt:
            return masks[0]
        
        boxes = np.array(resp.bounding_boxes, dtype=int).reshape((-1, 4)).tolist()
        print(boxes)
        box = min([box for box in boxes if box[1] >= 50], key=lambda box:box[1])
        idx = boxes.index(box)
        return masks[idx]

    def on_rec_img(self, msg:Image):
        if self.is_quit:
            return
        self.img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def on_rec_depth(self, msg:Image):
        if self.is_quit:
            return
        self.img_dep = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1") / 1000.0
        self.cld, _ = reconstruct(self.img_dep, self.cam_intrin)
        # hd = Header(frame_id=self.coord_cam, stamp=rospy.Time.now())
        # fds = [
        #     PointField('x', 0, PointField.FLOAT32, 1),
        #     PointField('y', 4, PointField.FLOAT32, 1),
        #     PointField('z', 8, PointField.FLOAT32, 1),
        # ]
        # msg_pc2 = pc2.create_cloud(hd, fds, np.reshape(self.cld, (-1, 3)))
        # self.pub_cld.publish(msg_pc2)
    
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

    def quit(self, code=None):
        self.is_quit = True
        exit(code)

if __name__ == '__main__':
    # reset_arm()
    fbi = FBI()