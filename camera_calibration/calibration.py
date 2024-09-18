from scipy.optimize import least_squares
import numpy as np
import cv2

# 加载数据
odometry_data = np.load('odometry_data.npy', allow_pickle=True)
tag_poses = np.load('tag_poses.npy', allow_pickle=True)

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion to a rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def solve_hand_eye(A, B):
    def residuals(x):
        X = np.array([[x[0], -x[3], x[2], x[4]],
                      [x[3], x[0], -x[1], x[5]],
                      [-x[2], x[1], x[0], x[6]],
                      [0, 0, 0, 1]])
        res = []
        for i in range(len(A)):
            res.append((A[i] @ X - X @ B[i]).ravel())
        return np.concatenate(res)

    x0 = np.array([1, 0, 0, 0, 0, 0, 0])
    res = least_squares(residuals, x0)
    return np.array([[res.x[0], -res.x[3], res.x[2], res.x[4]],
                     [res.x[3], res.x[0], -res.x[1], res.x[5]],
                     [-res.x[2], res.x[1], res.x[0], res.x[6]],
                     [0, 0, 0, 1]])

# 计算A和B矩阵
A = [np.eye(4) for _ in range(len(odometry_data))]
B = [np.eye(4) for _ in range(len(tag_poses))]

# 填充A和B矩阵
for i in range(len(odometry_data)):
    odom = odometry_data[i]
    tag_pose = tag_poses[i]
    # 生成A矩阵（相机位姿变换）
    position = [odom.translation.x, odom.translation.y, odom.translation.z]
    orientation = [odom.rotation.w, odom.rotation.x, odom.rotation.y, odom.rotation.z]
    rotation_matrix_odom = quaternion_to_rotation_matrix(orientation)
    A[i][:3, :3] = rotation_matrix_odom
    A[i][:3, 3] = position
    # 生成B矩阵（AprilTag位姿变换）
    B[i][:3, :3] = cv2.Rodrigues(tag_pose[0])[0]
    B[i][:3, 3] = tag_pose[1].flatten()

# 求解手眼标定矩阵
X = solve_hand_eye(A, B)
print("手眼标定矩阵：\n", X)
