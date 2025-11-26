#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo
from camera_info_manager import CameraInfoManager


def publisher():
    rospy.init_node("camera_info_publisher")

    # 初始化 camera_info_manager，传入相机名称和标定文件路径
    camera_info_url = "/home/locobot/locobot/src/custom_pkgs/locobot/config/realsense/d455_calibration.yaml"
    manager = CameraInfoManager(cname="camera", url=camera_info_url)

    # 确保加载成功
    if not manager.loadCameraInfo():
        rospy.logerr("Failed to load camera calibration file")
        return

    # 获取 camera_info
    camera_info_msg = manager.getCameraInfo()

    # 发布 camera_info 话题
    pub = rospy.Publisher("/locobot/camera/color/camera_info", CameraInfo, queue_size=10)

    rate = rospy.Rate(10)  # 发布频率

    while not rospy.is_shutdown():
        pub.publish(camera_info_msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
