# usage: init a SKillLib object with a Demo object (aka, robot interface) and a PoseEstimator object, then init a VLMPlanner object with the skill library
import os
import os.path as osp
import json
import rospy
import argparse
import textwrap

from geometry_msgs.msg import *
from openai import OpenAI

from demo import *
from pose_estimate import PoseEstimator

# global variables
content = None

# init global variables
path_gpose = osp.join(osp.dirname(osp.abspath(__file__)), "goal_pose.json")
with open(path_gpose, "r") as f:
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
        self.IND = " " * 4

        self.skill_lib = skill_lib

        self.system_descr = """
        You are a robotic task planner. Your job is to parse the given instruction to generate a list of commands from the robotic skill library (formatted as XML). Specifically, given skill library like:
        <skill_library>
            <skill name="OP1"> description1 </skill>
            <skill name="OP2"> description2 </skill>
            ...
        </skill_library>,
        the expected format of commands is 'OP obj; ...'

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
        skill_entries = [
            VLMPlanner.skill_temp.format(name, textwrap.indent(_dedent(func.__doc__), self.IND))
            for name, func in skill_lib.items()
        ]
        tmp = textwrap.indent("\n".join(skill_entries), self.IND)
        self.skills_descr = self.skills_descr.format(tmp)

        # task template
        self.task_temp = "The given task is: {}"

        # TODO: check if the formatting works as expected
        print("==============System description==============")
        print(self.system_descr)
        print("==============Skills description==============")
        print(self.skills_descr)
        print("==============================================")

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


class SkillLib:
    def __init__(self, robo_iface: Demo, pose_estim: PoseEstimator):
        self.robo = robo_iface
        self.estim = pose_estim

    def grasp(self, obj: str):
        """Grasp given object.
        Args:
            obj: str, supported list: [{}]
        Returns:
            log: str, execution log
        """
        log = f"Executing 'grasp {obj}'\n"
        try:
            self.estim.reset_obj(f"./GMatch/cache/{obj}.pt")
            errcode, errmsg = self.estim.run_once()
            if errcode != 0:
                raise RuntimeError(errmsg)
            rospy.sleep(0.1)
            log += "Pose estimation done.\n"

            goal_pose = content[obj]["goal_pose"]
            offset = content[obj]["offset"]
            log += f"Goal pose loaded ({goal_pose}).\n"
            p, q = goal_pose[:3], goal_pose[3:]
            # publish static transformation
            msg = TransformStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.robo.coord_targ
            msg.child_frame_id = self.robo.coord_ee_goal
            msg.transform.translation = Vector3(*p)
            msg.transform.rotation = Quaternion(*q)
            self.robo.tf_spub.sendTransform(msg)
            rospy.sleep(0.1)

            goal = transform_pose(self.robo.tf_buf, self.robo.coord_map, self.robo.coord_ee_goal)
            log += "Goal pose published and transformed.\n"

            self.robo.grasp_goal(goal, offset)
            log += "Grasp action executed.\n"
            self.robo.arm.move_joints(np.array([0.0, -1.1, 1.55, 0.0, -0.5, 0.0]))
            log += f"Grasping {obj} finished and move to ready state.\n"
        except Exception as e:
            e = textwrap.indent(str(e))
            log += f"Grasping {obj} failed with exception: \n{e}\n"
        return log

    def place(self, obj: str):
        """Place given object on target location.
        Args:
            obj: str, supported list: [{}]
        Returns:
            log: str, execution log
        """
        log = f"Executing 'place {obj}'\n"
        try:
            self.estim.reset_obj(f"./GMatch/cache/{obj}.pt")
            self.estim.run_once()
            rospy.sleep(0.1)
            log += "Pose estimation done.\n"

            goal_pose = content[obj]["goal_pose"]
            offset = content[obj]["offset"]
            p, q = goal_pose[:3], goal_pose[3:]
            log += f"Goal pose loaded ({goal_pose}).\n"

            # publish static transformation
            msg = TransformStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.robo.coord_targ
            msg.child_frame_id = self.robo.coord_ee_goal
            msg.transform.translation = Vector3(*p)
            msg.transform.rotation = Quaternion(*q)
            self.robo.tf_spub.sendTransform(msg)
            rospy.sleep(0.1)

            goal = transform_pose(self.robo.tf_buf, self.robo.coord_map, self.robo.coord_ee_goal)
            log += "Goal pose published and transformed.\n"

            self.robo.place_goal(goal, offset)
            log += "Place action executed.\n"
        except Exception as e:
            e = textwrap.indent(str(e))
            log += f"Placing {obj} failed with exception: \n{e}\n"
        return log

    def look(self, dir: str):
        """Look more in given direction (0.2 rad).
        Args:
            dir: str, supported list: [left, right, up, down]
        Returns:
            log: str, execution log
        """
        log = f"Executing 'look {dir}'\n"
        try:
            if dir not in ["left", "right", "up", "down"]:
                raise ValueError("Invalid direction argument. (supported: left, right, up, down)")
            c = self.robo.cam
            yaw1, pitch1 = c.yaw, c.pitch
            if dir in ["left", "right"]:
                yaw1 += 0.2 if dir == "left" else -0.2
            else:
                pitch1 += -0.2 if dir == "up" else 0.2
            if not (c.yaw_limits[0] <= yaw1 <= c.yaw_limits[1]):
                raise ValueError(f"Yaw {yaw1} out of limits {c.yaw_limits}!")
            if not (c.pitch_limits[0] <= pitch1 <= c.pitch_limits[1]):
                raise ValueError(f"Pitch {pitch1} out of limits {c.pitch_limits}!")
            c.turret.pan_tilt_move(pan_position=yaw1, tilt_position=pitch1)
            log += f"Look action executed. curr_yaw: {c.yaw}, curr_pitch: {c.pitch}; yaw limits: {c.yaw_limits}, pitch limits: {c.pitch_limits}\n"
        except Exception as e:
            e = textwrap.indent(str(e))
            log += f"Looking {dir} failed with exception: \n{e}\n"
        return log

    def move(self, dir: str):
        """Move robot in given direction by 0.2m. The orientation remains the same.
        Args:
            dir: str, supported list: [forward, backward, left, right]
        Returns:
            log: str, execution log
        """
        log = f"Executing 'move {dir}'\n"
        try:
            if dir not in ["forward", "backward", "left", "right"]:
                raise ValueError("Invalid direction argument. (supported: forward, backward, left, right)")
            x0, y0, theta0 = self.robo.chas.pose2d
            i = ["forward", "left", "backward", "right"].index(dir)
            theta1 = theta0 + i * (np.pi / 2)
            x1 = x0 + 0.2 * np.cos(theta1)
            y1 = y0 + 0.2 * np.sin(theta1)
            self.robo.chas.reach(x1, y1, theta=theta0)
            log += "Move action executed.\n"
        except Exception as e:
            e = textwrap.indent(str(e))
            log += f"Moving {dir} failed with exception: \n{e}\n"
        return log

    def turn(self, dir: str):
        """Turn robot to given direction (90 degrees, close-loop).
        Args:
            dir: str, supported list: [left, right]
        Returns:
            log: str, execution log
        """
        log = f"Executing 'turn {dir}'\n"
        try:
            if dir not in ["left", "right"]:
                raise ValueError("Invalid direction argument. (supported: left, right)")
            angle = np.pi / 2 if dir == "left" else -np.pi / 2
            self.robo.chas.rotate(angle)
            log += "Turn action executed.\n"
        except Exception as e:
            e = textwrap.indent(str(e))
            log += f"Turning {dir} failed with exception: \n{e}\n"
        return log

    # @staticmethod
    # def gsam(cmd): ...


grasp_objs = [key for key in content.keys() if content[key]["type"] == "grasp"]
place_objs = [key for key in content.keys() if content[key]["type"] == "place"]
SkillLib.grasp.__doc__ = SkillLib.grasp.__doc__.format(", ".join(grasp_objs))
SkillLib.place.__doc__ = SkillLib.place.__doc__.format(", ".join(place_objs))


if __name__ == "__main__":
    rospy.init_node("vlm")

    robo_iface = Demo()

    parser = argparse.ArgumentParser(description="VLM task planner that calls atomic skills")
    parser.add_argument("--minL", type=int, default=8, help="Min length of matches to use.")
    parser.add_argument("--debug", type=int, default=-1, help="Debug level, bigger means more info.")
    args = parser.parse_args()
    pose_estim = PoseEstimator(args)

    sl_obj = SkillLib(robo_iface, pose_estim)
    sl = {fo: getattr(sl_obj, fo) for fo in dir(SkillLib) if not fo.startswith("__")}

    planner = VLMPlanner(skill_lib=sl, local=False)
    breakpoint()
    cmds = planner.plan("come close to put gum bottle on the realsense box")

    for cmd in cmds:
        sl[cmd[0]]()
