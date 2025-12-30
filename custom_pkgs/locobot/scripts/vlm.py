import os
import json
import rospy
import argparse
import textwrap

from geometry_msgs.msg import *
from openai import OpenAI

from demo import *
from pose_estimate import PoseEstimator

# global variables
robo_iface = None
pose_estim = None
content = None

# init global variables
rospy.init_node("vlm")
robo_iface = Demo()
parser = argparse.ArgumentParser(description="VLM task planner that calls atomic skills")
parser.add_argument("--minL", type=int, default=-1, help="Debug level, bigger means more info.")
parser.add_argument("--debug", type=int, default=-1, help="Debug level, bigger means more info.")
args = parser.parse_args()
pose_estim = PoseEstimator(args)
with open("./goal_pose.json", "r") as f:
    content = json.load(f)


def _dedent(docstr: str) -> str:
    contents = docstr.strip().split("\n", 1)
    return contents[0] if len(contents) == 1 else contents[0] + "\n" + textwrap.dedent(contents[1])


class VLMPlanner:
    skill_temp = '<skill name="{}">\n{}\n</skill>'

    def __init__(self, skill_lib: dict, local: bool = False):
        """use doc str and func name in `skill_library` to build skills description"""
        # TODO: not sure formatting here works as expected. since \t and spaces may be mixed.
        self.context = ""

        self.skill_lib = skill_lib

        self.system_descr = """
        You are a robotic task planner. Your job is to parse the given instruction to generate a list of commands from the robotic skill library (formatted as XML).

        <skill_library>
        <skill name="OP1"> description1 </skill>
        <skill name="OP2"> description2 </skill>
        ...
        </skill_library>
        The expected format of commands is 'OP obj; ...'

        Do NOT add explanations, markdown, or extra fields.

        <example>
        Given the skill library with two skills:
        <skill_library>
            <skill name="grasp"> 
            grasp given object.
            Args:
                object: str, supported list: [bowl, coke_bottle, handle, pen]
            Returns:
                None
            </skill>
            <skill name="place">
            place given object on target location.
            Args:
                object: str, supported list: [bowl, cabinet, desk, book]
            Returns:
                None
            </skill>
        </skill_library>
        For task 'put the bottle on the desk', you try to first grasp the bottle and place it on the desk. And then you check that bottle (namely, coke_bottle) is in grasp list and desk in in place list. Finally, you return 'grasp bowl; place desk;'.
        For task 'put the book on the cabinet', you try to grasp the book then place it on the cabinet. But you find book is not in grasp list (even if it's in place list). Therefore, you should return '' instead of 'grasp book; place cabinet;'
        </example>
        """
        self.system_descr = _dedent(self.system_descr.strip())

        self.skills_descr = """
        Now, your skill library is:
        <skill_library>
            {}
        </skill_library>
        """
        self.skills_descr = _dedent(self.skills_descr.strip())
        skill_entries = [VLMPlanner.skill_temp.format(name, func.__doc__) for name, func in skill_lib.items()]
        tmp = textwrap.indent("\n".join(skill_entries), "\t")
        self.skills_descr = self.skills_descr.format(tmp)

        # task template
        self.task_temp = "The given task is: {}"

        # TODO: check if the formatting works as expected
        print("==============System description==============")
        print(self.system_descr)
        print("==============Skills description==============")
        print(self.skills_descr)
        print("===============================================")

        if local:
            # Locally Deployed Qwen API
            url = "http://v100:8000/v1"
            api_key = ""
            model_name = "Qwen/Qwen2.5-14B-Instruct"
        else:
            # Remote Qwen API
            url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            api_key = os.getenv("DASHSCOPE_API_KEY")
            model_name = "qwen-plus"

        self.client = OpenAI(api_key=api_key, base_url=url)
        self.model_name = model_name

    def plan(self, task_descr: str):
        """parse task description to commands list"""
        system_prompt = self.system_descr + "\n" + self.skills_descr
        user_prompt = self.task_temp.format(task_descr)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        resp = completion.model_dump_json()
        result = json.loads(resp)
        content = result["choices"][0]["message"]["content"]
        cmds = content.strip(";").split(";")
        cmds = list(map(lambda x: x.strip().split(maxsplit=1), cmds))
        return cmds

    def exec(self, cmd):
        """exec command, judge success with obs and replan automatically"""
        pass


grasp_objs = [key for key in content.keys() if content[key]["type"] == "grasp"]
place_objs = [key for key in content.keys() if content[key]["type"] == "place"]


class SkillLib:
    @staticmethod
    def grasp(obj: str):
        f"""Grasp given object.
        Args:
            obj: str, supported list: {grasp_objs}
        Returns:
            log: str, execution log
        """
        log = f"Executing 'grasp {obj}'"
        try:
            pose_estim.reset_obj(f"./GMatch/cache/{obj}.pt")
            pose_estim.run_once()
            rospy.sleep(0.1)
            log += "\nPose estimation done."

            goal_pose = content[obj]["goal_pose"]
            offset = content[obj]["offset"]
            log += f"\nGoal pose loaded ({goal_pose})."
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
            log += "\nGoal pose published and transformed."

            robo_iface.grasp_goal(goal, offset)
            log += "\nGrasp action executed."
            robo_iface.arm.move_joints(np.array([0.0, -1.1, 1.55, 0.0, -0.5, 0.0]))
            log += f"\nGrasping {obj} finished and move to ready state."
        except Exception as e:
            log += f"\nGrasping {obj} failed with exception: {e}"
        return log

    @staticmethod
    def place(obj: str):
        f"""Place given object on target location.
        Args:
            obj: str, supported list: {place_objs}
        Returns:
            log: str, execution log
        """
        log = f"Executing 'place {obj}'"
        try:
            pose_estim.reset_obj(f"./GMatch/cache/{obj}.pt")
            pose_estim.run_once()
            rospy.sleep(0.1)
            log += "\nPose estimation done."

            goal_pose = content[obj]["goal_pose"]
            offset = content[obj]["offset"]
            p, q = goal_pose[:3], goal_pose[3:]
            log += f"\nGoal pose loaded ({goal_pose})."

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
            log += "\nGoal pose published and transformed."

            robo_iface.place_goal(goal, offset)
            log += "\nPlace action executed."
        except Exception as e:
            log += f"\nPlacing {obj} failed with exception: {e}"
        return log

    @staticmethod
    def look(dir: str):
        """Look more in given direction (0.2 rad).
        Args:
            dir: str, supported list: [left, right, up, down]
        Returns:
            log: str, execution log
        """
        log = f"Executing 'look {dir}'"
        try:
            if dir not in ["left", "right", "up", "down"]:
                raise ValueError("Invalid direction argument. (supported: left, right, up, down)")
            c = robo_iface.cam
            if dir == "left":
                robo_iface.cam.turret.pan(c.curr_yaw + 0.2)
            elif dir == "right":
                robo_iface.cam.turret.pan(c.curr_yaw - 0.2)
            elif dir == "up":
                robo_iface.cam.turret.tilt(c.curr_pitch - 0.2)
            elif dir == "down":
                robo_iface.cam.turret.tilt(c.curr_pitch + 0.2)
            log += f"Look action executed. curr_yaw: {c.curr_yaw}, curr_pitch: {c.curr_pitch}; yaw limits: {c.turret.yaw_limits}, pitch limits: {c.turret.pitch_limits}"
        except Exception as e:
            log += f"\nLooking {dir} failed with exception: {e}"
        return log

    @staticmethod
    def move(dir: str):
        """Move robot in given direction (0.3 m/s by 1 second, open-loop).
        Args:
            dir: str, supported list: [forward, backward, left, right]
        Returns:
            log: str, execution log
        """
        log = f"Executing 'move {dir}'"
        try:
            v = 0.3
            msg = Twist()
            if dir not in ["forward", "backward", "left", "right"]:
                raise ValueError("Invalid direction argument. (supported: forward, backward, left, right)")
            if dir in ["forward", "backward"]:
                sgn = 1 if dir == "forward" else -1
                msg.linear.x = sgn * v
                robo_iface.chas.pub_vel.publish(msg)
                rospy.sleep(1)
                robo_iface.chas.pub_vel.publish(Twist())
            else:
                log += f"\nRotating {dir} first."
                sgn = 1 if dir == "left" else -1
                robo_iface.chas.rotate(sgn * np.pi / 2)
                log += "\nMoving forward."
                msg.linear.x = v
                robo_iface.chas.pub_vel.publish(msg)
                rospy.sleep(1)
                robo_iface.chas.pub_vel.publish(Twist())
                log += "\nRotating back."
                robo_iface.chas.rotate(-sgn * np.pi / 2)
            log += "Move action executed."
        except Exception as e:
            log += f"\nMoving {dir} failed with exception: {e}"
        return log

    @staticmethod
    def turn(dir: str):
        """Turn robot in given direction (90 degrees, close-loop).
        Args:
            dir: str, supported list: [left, right]
        Returns:
            log: str, execution log
        """
        log = f"Executing 'turn {dir}'"
        try:
            if dir not in ["left", "right"]:
                raise ValueError("Invalid direction argument. (supported: left, right)")
            angle = np.pi / 2 if dir == "left" else -np.pi / 2
            robo_iface.chas.rotate(angle)
            log += "Turn action executed."
        except Exception as e:
            log += f"\nTurning {dir} failed with exception: {e}"
        return log

    # @staticmethod
    # def gsam(cmd): ...

if __name__ == "__main__":
    # skill library in dictionary
    sl = {fo: getattr(SkillLib, fo) for fo in dir(SkillLib) if not fo.startswith("__")}

    planner = VLMPlanner(skill_lib=sl, local=False)
    cmds = planner.plan("put gum bottle on the realsense box")

    for cmd in cmds:
        sl[cmd[0]]()
