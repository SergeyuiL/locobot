import os
import json
import rospy
import argparse

from geometry_msgs.msg import *
from openai import OpenAI

from demo import *
from pose_estimate import PoseEstimator


# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# template = """
# You are a robotic task planner. Your job is to extract the **source object** (to be grasped) and the **target container/object** (to place into/on) from a high-level instruction.

# You have two operations available:
# 1. grasp <source_object>: to pick up the source object.
# 2. place <target_object>: to put down the source object into/on the target object.

# Given a task description, respond with an ordered list of the operation and its argument, formatted as follows:
# '<OP> <obj1>; <OP> <obj2>; ...'
# where <OP> is either 'grasp' or 'place'

# Do NOT add explanations, markdown, or extra fields. In case that either source object or target object is not supported, return 'None'.

# E.g. Given skill library
#     <grasp>bowl, bottle, handle, pen</grasp>
#     <place>bowl, cabinet, desk, book</place>
# for task 'put the pen on the bottle', the response should be: 'None' because 'bottle' is not a valid target object.
# for task 'put the bowl on the desk', the response should be: 'grasp bowl; place desk'
# for task 'make up the desk so that the pen is on the book and the book is on the desk', the response should be: 'grasp book; place desk; grasp pen; place book'

# Now, your skill library is:
# <grasp>{}</grasp>
# <place>{}</place>
# And you need to parse this task: {}
# """

# grasp_targets = "bowl, bottle, handle, pen"
# place_targets = "bowl, cabinet, desk, book"
# task_descr = "put the pen in the bowl, then put the bowl on the desk"

# prompt = template.format(grasp_targets, place_targets, task_descr)

# completion = client.chat.completions.create(
#     # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     model="qwen-plus",
#     messages=[{"role": "user", "content": prompt}],
# )
# resp = completion.model_dump_json()
# result = json.loads(resp)
# content = result["choices"][0]["message"]["content"]
# print(content)

parser = argparse.ArgumentParser(description="VLM task planner that calls atomic skills")
parser.add_argument("--minL", type=int, default=-1, help="Debug level, bigger means more info.")
parser.add_argument("--debug", type=int, default=-1, help="Debug level, bigger means more info.")
args = parser.parse_args()

rospy.init_node("vlm")

d = Demo()
pe = PoseEstimator(args)

content = "grasp gum_bottle; place realsensebox"
cmds = content.split(";")
cmds = list(map(lambda x: x.strip().split(), cmds))

with open("./goal_pose.json", "r") as f:
    content = json.load(f)

tmp = None
for cmd in cmds:
    op, obj = cmd
    print(f"exec '{op} {obj}'")
    pe.reset_obj(f'./GMatch/cache/{obj}.pt')
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

    print(goal)

    d.authenticate(f"{op}")

    if op == "grasp":
        d.grasp_goal(goal, offset)
        d.arm.move_joints(np.array([0.0, -1.1, 1.55, 0.0, -0.5, 0.0]))
    else:
        d.place_goal(goal, offset)
