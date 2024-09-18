import cv2
import numpy as np
from apriltag import apriltag

images = np.load('images.npy', allow_pickle=True)
tag_size = 0.036  # AprilTag的边长，单位为米
K = np.array([
    [384.3591003417969, 0.0, 319.05145263671875],
    [0.0, 383.3277893066406, 244.51524353027344],
    [0.0, 0.0, 1.0]
])
D = np.array([-0.05430610105395317, 0.06399306654930115, -0.0011168108321726322, 4.981948950444348e-05, -0.02058405615389347])

tag_poses = []
detector = apriltag("tagStandard41h12")
for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    
    for result in results:
        # 计算tag位姿
        object_points = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]]) * tag_size
        image_points = np.array(result['lb-rb-rt-lt']).reshape(-1, 2)
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, K, D)
        if retval:
            tag_poses.append((rvec, tvec))
        else:
            print("Failed to compute the pose of the tag.")

# 保存tag位姿数据到npy文件
np.save('tag_poses.npy', tag_poses)
print("Tag poses saved to tag_poses.npy")

