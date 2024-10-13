import rospy
from interbotix_xs_modules.core import InterbotixRobotXSCore
from interbotix_xs_modules.turret import InterbotixTurretXSInterface
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse

class LocoboCamera:
    def __init__(self) -> None:
        dxl = InterbotixRobotXSCore(
                                    robot_model="wx250s",
                                    robot_name="locobot",
                                    init_node=False,
                                    )
        
        self.camera = InterbotixTurretXSInterface(core=dxl, group_name="camera",
                                                  pan_profile_velocity=4.0, pan_profile_acceleration=0.6,
                                                  tilt_profile_velocity=4.0, tilt_profile_acceleration=0.6)

        self.init_pitch = 0.9
        self.pitch_limits = [0.5, -0.1] # positive is down, negetive is up
        self.yaw_limits = [-0.7, 0.7]
        self.pitch_server = rospy.Service("/locobot/camera_pitch_control", SetBool, self.pitch_scan)
        self.yaw_server = rospy.Service("/locobot/camera_yaw_control", SetBool, self.yaw_scan)
        self.reset_camera()

    def reset_camera(self):
        self.camera.pan_tilt_move(pan_position=0.0, tilt_position=self.init_pitch,
                                  pan_profile_velocity=2.0, pan_profile_acceleration=0.3, 
                                  tilt_profile_velocity=2.0, tilt_profile_acceleration=0.3)
        # self.camera.pan_tilt_go_home()
        # self.pitch_control(self.init_pitch)
    
    def pitch_control(self, position):
        # desired goal position [rad]
        self.camera.tilt(position)
    
    def yaw_control(self, position):
        # desired goal position [rad]
        self.camera.pan(position)
    
    def pitch_scan(self, req: SetBoolRequest):
        if req.data:
            self.pitch_control(self.pitch_limits[0])
            self.pitch_control(self.pitch_limits[1])
            self.reset_camera()
        return SetBoolResponse(True, "pitch scan completed!")
    
    def yaw_scan(self, req: SetBoolRequest):
        if req.data:
            self.yaw_control(self.yaw_limits[0])
            self.yaw_control(self.yaw_limits[1])
            self.reset_camera()
        return SetBoolResponse(True, "yaw scan completed!")


if __name__ == "__main__":
    rospy.init_node("test_camera_control")
    camera = LocoboCamera()
    camera.reset_camera()
    camera.pitch_control(0.4)
    camera.pitch_control(-0.25)
    camera.reset_camera()
    # camera.yaw_control(0.18)
    # camera.yaw_control(-0.18)
    # camera.reset_camera()
    rospy.spin()