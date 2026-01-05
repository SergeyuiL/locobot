#!/usr/bin/env python3

import rospy
from interbotix_xs_modules.core import InterbotixRobotXSCore
from interbotix_xs_modules.turret import InterbotixTurretXSInterface
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from locobot.srv import SetFloat32, SetFloat32Request, SetFloat32Response


class LocobotCamera:
    """provide services to adjust the camera pitch and yaw (currently scanning, maybe add angle control later)"""

    def __init__(self, serving=True) -> None:
        dxl = InterbotixRobotXSCore(
            robot_model="wx250s",
            robot_name="locobot",
            init_node=False,
        )

        self.turret = InterbotixTurretXSInterface(
            core=dxl,
            group_name="camera",
            pan_profile_velocity=4.0,
            pan_profile_acceleration=0.6,
            tilt_profile_velocity=4.0,
            tilt_profile_acceleration=0.6,
        )

        self.init_pitch = 0.6
        self.init_yaw = 0.0

        # pitch_config = [-0.1, 0.9]
        # yaw_config = [-0.7, 0.7]
        pitch_config = self.turret.info[self.turret.pan_name]  # positive is down, negetive is up
        yaw_config = self.turret.info[self.turret.tilt_name]  # positive is left, negetive is right
        self.pitch_limits = (pitch_config["lower_limit"], pitch_config["upper_limit"])
        self.yaw_limits = (yaw_config["lower_limit"], yaw_config["upper_limit"])

        if serving:
            # for debugging in terminal
            self.pitch_server = rospy.Service("/locobot/camera_pitch_control", SetFloat32, self.on_rec_pitch)
            self.yaw_server = rospy.Service("/locobot/camera_yaw_control", SetFloat32, self.on_rec_yaw)
        self.reset_camera()

    def reset_camera(self):
        self.turret.pan_tilt_move(pan_position=self.init_yaw, tilt_position=self.init_pitch)

    @property
    def yaw(self):
        return self.turret.get_joint_commands()[0]

    @property
    def pitch(self):
        return self.turret.get_joint_commands()[1]

    def on_rec_pitch(self, req: SetFloat32Request):
        self.turret.tilt(req.data)
        res = self.pitch_limits[0] <= req.data <= self.pitch_limits[1]
        mes = f"{'Succeed' if res else 'Failed'} to pitch {req.data:.2f} rad! limit: {self.pitch_limits}"
        return SetFloat32Response(res, mes)

    def on_rec_yaw(self, req: SetFloat32Request):
        self.turret.pan(req.data)
        res = self.yaw_limits[0] <= req.data <= self.yaw_limits[1]
        mes = f"{'Succeed' if res else 'Failed'} to yaw {req.data:.2f} rad! limit: {self.yaw_limits}"
        return SetFloat32Response(res, mes)

    def pitch_scan(self, req: SetBoolRequest):
        if req.data:
            self.turret.tilt(self.pitch_limits[0])
            self.turret.tilt(self.pitch_limits[1])
            self.reset_camera()
        return SetBoolResponse(True, "pitch scan completed!")

    def yaw_scan(self, req: SetBoolRequest):
        if req.data:
            self.turret.pan(self.yaw_limits[0])
            self.turret.pan(self.yaw_limits[1])
            self.reset_camera()
        return SetBoolResponse(True, "yaw scan completed!")


if __name__ == "__main__":
    rospy.init_node("test_camera_control")
    lc = LocobotCamera()
    lc.reset_camera()
    lc.turret.tilt(0.4)
    lc.turret.tilt(-0.25)
    lc.reset_camera()
    # lc.turret.pan(0.18)
    # lc.turret.pan(-0.18)
    # lc.reset_camera()
    rospy.spin()
