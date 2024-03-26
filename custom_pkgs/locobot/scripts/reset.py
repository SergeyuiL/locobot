from interbotix_xs_modules.locobot import InterbotixLocobotXS


def main():
    locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s", use_move_base_action=True)
    
    locobot.gripper.close()
    locobot.arm.go_to_sleep_pose()

    # locobot.camera.pan_tilt_go_home()
    locobot.camera.pan_tilt_move(0, 0.2618)

if __name__=='__main__':
    main()
