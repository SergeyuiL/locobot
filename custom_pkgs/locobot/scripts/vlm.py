import os
import json
import rospy
import argparse

from geometry_msgs.msg import *
from openai import OpenAI

from demo import *
from pose_estimate import PoseEstimator

robo_iface = None
pose_estim = None


def vlm_plan(task_descr, local=False):
    if local:
        # Locally Deployed Qwen API
        url = "http://v100:8000/v1"
        api_key = ""
        model_name = "Qwen/Qwen2.5-14B-Instruct"
    else:
        # Remote Qwen API
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = (os.getenv("DASHSCOPE_API_KEY"),)
        model_name = "qwen-plus"

    client = OpenAI(api_key=api_key, base_url=url)

    template = """
    You are a robotic task planner. Your job is to parse the given instruction to generate a list of commands from the robotic skill library.
    <skill_library>
        <OP1> obj1, ... </OP1>
        <OP2> obj2, ... </OP2>
        ...
    </skill_library>
    The expected list of commands is 'OP obj1; OP obj2; ...'

    Do NOT add explanations, markdown, or extra fields. In case that either OP is not supported or its operated object is not supported, return 'None'. Here is an example.

    <example>
    Given skill library:
    <skill_library>
        <grasp>bowl, coke_bottle, handle, pen</grasp>
        <place>bowl, cabinet, desk, book</place>
    </skill_library>
    For task 'put the bottle on the desk', you try to first grasp the bottle and place it on the desk. And then you check that bottle (namely, coke_bottle) is in grasp list and desk in in place list. Finally, you return 'grasp bowl; place desk;'.
    For task 'put the book on the cabinet', you try to grasp the book then place it on the cabinet. But you find book is not in grasp list (even if it's in place list). Therefore, you should return 'None' instead of 'grasp book; place cabinet;'
    </example>

    Now, your skill library is:
    <skill_library>
        <grasp>{}</grasp>
        <place>{}</place>
        <move>forward, backward</move>
        <turn>left, right</turn>
        <look>left, right, up, down</look>
    </skill_library>
    And you need to parse this task: {}
    """

    grasp_targets = "cabinet, coffee_bottle, gum_bottle"
    place_targets = "sugar_jar, toybox, realsensebox"

    prompt = template.format(grasp_targets, place_targets, task_descr)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    resp = completion.model_dump_json()
    result = json.loads(resp)
    content = result["choices"][0]["message"]["content"]
    cmds = content.split(";")
    cmds = list(map(lambda x: x.strip().split(maxsplit=1), cmds))
    return cmds


class CmdExec:
    @staticmethod
    def manipulate(cmd):
        op, obj = cmd
        pose_estim.reset_obj(f"./GMatch/cache/{obj}.pt")
        pose_estim.run_once()
        rospy.sleep(0.1)

        goal_pose = content[obj]["goal_pose"]
        offset = content[obj]["offset"]
        p, q = goal_pose[:3], goal_pose[3:]

        # publish static transformation
        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = robo_iface.coord_targ
        msg.child_frame_id = robo_iface.coord_ee_goal
        msg.transform.translation = Vector3(*p)
        msg.transform.rotation = Quaternion(*q)
        robo_iface.tf_spub.sendTransform(msg)
        rospy.sleep(0.1)

        goal = transform_pose(robo_iface.coord_map, robo_iface.coord_ee_goal)

        if op == "grasp":
            robo_iface.grasp_goal(goal, offset)
            robo_iface.arm.move_joints(np.array([0.0, -1.1, 1.55, 0.0, -0.5, 0.0]))
        elif op == "place":
            robo_iface.place_goal(goal, offset)

    @staticmethod
    def look(cmd):
        _, arg = cmd
        assert arg in ["left", "right", "up", "down"]
        curr_yaw, curr_pitch = robo_iface.cam.curr_yaw, robo_iface.cam.curr_pitch
        if arg == "left":
            robo_iface.cam.turret.pan(curr_yaw + 0.2)
        elif arg == "right":
            robo_iface.cam.turret.pan(curr_yaw - 0.2)
        elif arg == "up":
            robo_iface.cam.turret.tilt(curr_pitch - 0.2)
        elif arg == "down":
            robo_iface.cam.turret.tilt(curr_pitch + 0.2)

    @staticmethod
    def move(cmd):
        _, arg = cmd
        assert arg in ["forward", "backward"]
        v = 0.5 if arg == "forward" else -0.5
        msg = Twist(Vector3(v, 0, 0), Vector3())
        robo_iface.chassis_vel.publish(msg)
        rospy.sleep(0.4)
        msg.linear.x = 0.0
        robo_iface.chassis_vel.publish(msg)

    @staticmethod
    def turn(cmd):
        _, arg = cmd
        assert arg in ["left", "right"]
        w = 1.0 if arg == "left" else -1.0
        msg = Twist(Vector3(), Vector3(0, 0, w))
        robo_iface.chassis_vel.publish(msg)
        rospy.sleep(0.5)
        msg.angular.z = 0.0
        robo_iface.chassis_vel.publish(msg)

    @staticmethod
    def gsam(cmd): ...


skill_lib = {
    "grasp": CmdExec.manipulate,
    "place": CmdExec.manipulate,
    "look": CmdExec.look,
    "move": CmdExec.move,
    "turn": CmdExec.turn,
    "gsam": CmdExec.gsam,
}


if __name__ == "__main__":
    rospy.init_node("vlm")

    robo_iface = Demo()

    parser = argparse.ArgumentParser(description="VLM task planner that calls atomic skills")
    parser.add_argument("--minL", type=int, default=-1, help="Debug level, bigger means more info.")
    parser.add_argument("--debug", type=int, default=-1, help="Debug level, bigger means more info.")
    args = parser.parse_args()
    pose_estim = PoseEstimator(args)

    cmds = vlm_plan("put gum bottle on the realsense box")

    with open("./goal_pose.json", "r") as f:
        content = json.load(f)

    for cmd in cmds:
        skill_lib[cmd[0]]()
