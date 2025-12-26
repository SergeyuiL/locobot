import os
import json
import rospy
import argparse

from geometry_msgs.msg import *
from openai import OpenAI

from demo import *
from pose_estimate import PoseEstimator


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


parser = argparse.ArgumentParser(description="VLM task planner that calls atomic skills")
parser.add_argument("--minL", type=int, default=-1, help="Debug level, bigger means more info.")
parser.add_argument("--debug", type=int, default=-1, help="Debug level, bigger means more info.")
args = parser.parse_args()

rospy.init_node("vlm")

d = Demo()
pe = PoseEstimator(args)

cmds = vlm_plan("put gum bottle on the realsense box")

with open("./goal_pose.json", "r") as f:
    content = json.load(f)

tmp = None
for cmd in cmds:
    op, obj = cmd
    print(f"exec '{op} {obj}'")
    pe.reset_obj(f"./GMatch/cache/{obj}.pt")
    pe.run_once()
    rospy.sleep(0.1)

    goal_pose = content[obj]["goal_pose"]
    offset = content[obj]["offset"]
    p, q = goal_pose[:3], goal_pose[3:]

    # publish static transformation
    msg = TransformStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = d.coord_targ
    msg.child_frame_id = d.coord_ee_goal
    msg.transform.translation = Vector3(*p)
    msg.transform.rotation = Quaternion(*q)
    d.tf_spub.sendTransform(msg)
    rospy.sleep(0.1)

    goal = transform_pose(
        Pose(position=Point(0, 0, 0), orientation=Quaternion(0, 0, 0, 1)),
        d.coord_map,
        d.coord_ee_goal,
        d.tf_buf,
    )

    if op == "grasp":
        d.grasp_goal(goal, offset)
        d.arm.move_joints(np.array([0.0, -1.1, 1.55, 0.0, -0.5, 0.0]))
    elif op == "place":
        d.place_goal(goal, offset)
