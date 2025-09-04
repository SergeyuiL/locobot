import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import yaml
from tqdm import tqdm
import shutil
import cv2 
from PIL import Image
import open3d as o3d
import time
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

from ultralytics import YOLO
from ultralytics import settings
            
class MapBuilder:        
    def __init__(self, data_dir, trans_camera_base, quat_camera_base, map2base_link_origin, is_map_origin=True):
        self.depth_dir = os.path.join(data_dir, "depth")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.pose_path = os.path.join(data_dir, "node_pose.txt")
        self.camera_info_path = os.path.join(data_dir, "camera_info.json")
        self.seg_dir = os.path.join(data_dir, "segment")
        
        self.rgb_pointcloud_save_path = os.path.join(data_dir, "rgb_pointcloud.ply")
        self.seg_pointcloud_save_path = os.path.join(data_dir, "seg_pointcloud.ply")
        self.object_info_path = os.path.join(data_dir, "object_info.json")
        self.map_pgm = os.path.join(data_dir, "map.pgm")
        self.map_yaml = os.path.join(data_dir, "map.yaml")
        
        r = R.from_quat(quat_camera_base)
        rot_matrix_camera2base = r.as_matrix()
        self.T_base_camera = np.eye(4)
        self.T_base_camera[:3, :3] = rot_matrix_camera2base
        self.T_base_camera[:3, 3] = trans_camera_base
        
        self.is_map_origin = is_map_origin
        if not is_map_origin:
            self.map2base_link_origin = map2base_link_origin

        # self.depth_denoise_kernel_size = depth_denoise_kernel_size
        self.camera_intrinsics = np.zeros((1, 4))
        self.camera_matrix = np.eye(3)
        self.rgb_list = []
        self.depth_list = []
        self.seg_list = []
        self.pose_list = []
        
        self.merge_time = None
        self.seg_time = None
        
        self.ckpt_dir = "/home/nvidia/skillsets_ws/src/skill_sets/semantic_map_builder/checkpoint"
        pt_path = os.path.join(self.ckpt_dir, "best.pt")
        engine_path = os.path.join(self.ckpt_dir, "best.engine")
        self.raw_model = YOLO(pt_path, task='segment')
        self.model = YOLO(engine_path, task='segment')
        self.class_names = self.raw_model.names
        # pt_path = os.path.join(self.ckpt_dir, "yolov8x-seg.pt")
        # engine_path = os.path.join(self.ckpt_dir, "yolov8x-seg.engine")
        # if not os.path.exists(self.ckpt_dir):
        #     os.makedirs(self.ckpt_dir)
        # settings.update({"runs_dir": data_dir, "weights_dir": self.ckpt_dir})
        
        # if not os.path.exists(engine_path):
        #     if not os.path.exists(pt_path):
        #         self.raw_model = YOLO("yolov8x-seg.pt")
        #     else:
        #         self.raw_model = YOLO(pt_path)
        #     self.raw_model.export(format="engine", dynamic=True, batch=1) 
        #     self.model = YOLO("yolov8x-seg.engine")
        # else:
        #     self.raw_model = YOLO(pt_path)
        #     self.model = YOLO(engine_path)
        # self.class_names = self.raw_model.names
        
        
        # self.model = YOLO(pt_path)
        # self.class_names = self.model.names
        
        
    def clear_seg_dir(self):
        if not os.path.exists(self.seg_dir):
            print(f"The directory {self.seg_dir} does not exist.")
            return
        
        for filename in os.listdir(self.seg_dir):
            file_path = os.path.join(self.seg_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        
    def data_load(self):
        # get pose
        base_poses = np.genfromtxt(self.pose_path, delimiter=' ')[:, 1:]
        indices = np.arange(base_poses.shape[0])
        for index in indices:  
            x_base, y_base, yaw_base = base_poses[index]
            T_map_base = np.array([
                                    [np.cos(yaw_base), -np.sin(yaw_base), 0, x_base],
                                    [np.sin(yaw_base), np.cos(yaw_base),  0, y_base],
                                    [0,                0,                 1, 0],
                                    [0,                0,                 0, 1]
                                ])
            T_map_camera = np.dot(T_map_base, self.T_base_camera)
            if self.is_map_origin:
                self.pose_list.append(T_map_camera)
            else:
                T_map_camera_origin = np.dot(self.map2base_link_origin, T_map_camera)
                self.pose_list.append(T_map_camera_origin)
        # get rgb path
        self.rgb_list = [os.path.join(self.rgb_dir, f"rgb_{index}.png") for index in indices]
        # get depth path
        self.depth_list = [os.path.join(self.depth_dir, f"depth_{index}.png") for index in indices]
        # get segment path
        self.seg_list = [os.path.join(self.seg_dir, f"rgb_{index}") for index in indices]
        # get camera info
        with open(self.camera_info_path, 'r') as file:
            camera_info = json.load(file)
        K = camera_info['K']
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        self.camera_intrinsics = np.array([fx, fy, cx, cy])
        self.camera_matrix = np.array([
                                [fx, 0,  cx],
                                [0,  fy, cy],
                                [0,  0,  1]
                            ])
        
    def load_map(self):
        image = Image.open(self.map_pgm)
        image = np.array(image)
        with open(self.map_yaml, 'r') as f:
            metadata = yaml.safe_load(f)
        return image, metadata
        
    def depth_denoise(self, depth_path, kernel_size=7):
        orin_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened_image = cv2.morphologyEx(orin_image, cv2.MORPH_OPEN, kernel)
        denoised_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
        return denoised_image

    def seg_rgb(self, confidence):
        seg_start_time = time.time()
        self.clear_seg_dir()
        pbar = tqdm(total=len(self.rgb_list))
        for rgb_path in self.rgb_list:
            rgb_img = cv2.imread(rgb_path)
            self.model.predict(source=rgb_img, save=True, save_txt=True, conf=confidence, project=self.seg_dir, name=os.path.splitext(os.path.basename(rgb_path))[0])
            pbar.update(1)
        seg_end_time = time.time()
        self.seg_time = seg_end_time - seg_start_time
    # def seg_rgb(self, confidence):
    #     def batch_generator(data, batch_size):
    #         for i in range(0, len(data), batch_size):
    #             yield data[i:i + batch_size]
        
    #     seg_start_time = time.time()
    #     self.clear_seg_dir()
    #     batch_size = 16
    #     pbar = tqdm(total=len(self.rgb_list))

    #     for batch_rgb_paths in batch_generator(self.rgb_list, batch_size):
    #         batch_rgb_images = [cv2.imread(rgb_path) for rgb_path in batch_rgb_paths]
            
    #         save_dirs = [self.seg_dir] * len(batch_rgb_paths)
    #         save_names = [os.path.splitext(os.path.basename(rgb_path))[0] for rgb_path in batch_rgb_paths]
            
    #         self.model.predict(source=batch_rgb_images, save=True, save_txt=True, conf=confidence, project=save_dirs, name=save_names)
            
    #         pbar.update(len(batch_rgb_paths))
    #     pbar.close()
    #     seg_end_time = time.time()
    #     self.seg_time = seg_end_time - seg_start_time
                
    def depth_to_pointcloud(self, depth_image):
        h, w = depth_image.shape
        fx, fy, cx, cy = self.camera_intrinsics
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        z = depth_image / 1000.0  
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy
        return np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    def transform_pointcloud(self, pointcloud, pose):
        R_mat = pose[:3, :3]
        t = pose[:3, 3]
        return (R_mat @ pointcloud.T).T + t
        
    def build_2dmap(self, max_depth=6000, DBSCAN_eps=0.05, DBSCAN_min_samples=50, voxel_size=0.1, kernel_size=7):
        class_point_clouds = {}
        all_class_ids = []
        mass_classes = ['storage basket', 'door', 'door handle']
        
        print("Start to merge 2D contours")
        cluster_2d_start_time = time.time()
        pbar = tqdm(total=len(self.rgb_list))
        for depth_path, seg_dir, pose in zip(self.depth_list, self.seg_list, self.pose_list):
            label_path = os.path.join(seg_dir, "labels/image0.txt")
            if not os.path.exists(label_path):
                print(f"Label file not found for {label_path}, skipping.")
                pbar.update(1)
                continue
            try:
                # depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth_image = self.depth_denoise(depth_path, kernel_size)
                seg_image = cv2.imread(os.path.join(seg_dir, "image0.jpg"))
            except Exception as e:
                print(f"Failed to load source data: {e}")
                pbar.update(1)
                continue  
            with open(label_path, 'r') as f:
                labels = f.readlines()

            point_cloud = self.depth_to_pointcloud(depth_image)
            transformed_point_cloud = self.transform_pointcloud(point_cloud, pose)
            depth_mask = (depth_image > 0) & (depth_image < max_depth)

            for label in labels:
                label_data = list(map(float, label.split()))
                class_id = int(label_data[0])
                points = np.array(label_data[1:]).reshape(-1, 2)

                class_name = self.class_names[class_id]
                if class_name in mass_classes:
                    # print(f"Get mass class:{class_name}, continue!")
                    continue
                if class_name not in class_point_clouds:
                    class_point_clouds[class_name] = o3d.geometry.PointCloud()
                points[:, 0] *= seg_image.shape[1]
                points[:, 1] *= seg_image.shape[0]
                points = points.astype(int)
                seg_mask = np.zeros(seg_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(seg_mask, [points], 1)
                seg_mask = seg_mask.flatten()
                valid_indices = depth_mask.flatten() & (seg_mask > 0)
                obj_points = transformed_point_cloud[valid_indices]
                obj_colors = seg_image.reshape(-1, 3)[valid_indices] / 255.0
                single_point_cloud = o3d.geometry.PointCloud()
                single_point_cloud.points = o3d.utility.Vector3dVector(obj_points)
                single_point_cloud.colors = o3d.utility.Vector3dVector(obj_colors)
                # Apply voxel downsampling
                single_point_cloud = single_point_cloud.voxel_down_sample(voxel_size)
                
                class_point_clouds[class_name] += single_point_cloud
                all_class_ids.extend([class_id] * len(obj_points))
            pbar.update(1)
        pbar.close()
        print("Start to cluster 2D points")
        
        final_object_info = {}
        for class_name, point_cloud in class_point_clouds.items():
            points = np.asarray(point_cloud.points)
            if len(points) == 0:
                continue
            # Project points to 2D plane
            points_2d = points[:, [0, 1]]
            # Perform DBSCAN clustering
            db = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(points_2d)
            labels = db.labels_
            unique_labels = np.unique(labels)
            contours = []
            for label in unique_labels:
                if label == -1:
                    continue
                mask = (labels == label)
                cluster_points = points_2d[mask]
                hull = cv2.convexHull(cluster_points.astype(np.float32))
                contours.append(hull[:, 0, :].tolist())
            if len(contours) == 0:
                continue
            else:
                final_object_info[class_name] = contours
        cluster_2d_end_time = time.time()
        self.merge_time = cluster_2d_end_time - cluster_2d_start_time
        print(f"Finish semantic map building\n Time consumption for Segmentation: {self.seg_time}, Merge points: {self.merge_time}")
        return final_object_info
        
    def merge_pointcloud(self, max_depth=6000, kernel_size=7):
        seg_point_cloud = o3d.geometry.PointCloud()
        rgb_point_cloud = o3d.geometry.PointCloud()

        pbar = tqdm(total=len(self.rgb_list))
        print("Start to merge pointcloud")
        start_time = time.time()
        for depth_path, rgb_path, seg_dir, pose in zip(self.depth_list, self.rgb_list, self.seg_list, self.pose_list):
            label_path = os.path.join(seg_dir, "label.txt")
            if not os.path.exists(label_path):
                print(f"Label file not found for {label_path}, skipping.")
                pbar.update(1)
                continue

            try:
                depth_image = self.depth_denoise(depth_path, kernel_size)
                rgb_image = cv2.imread(rgb_path)
                seg_image = cv2.imread(os.path.join(seg_dir, "rgb.jpg"))
            except Exception as e:
                print(f"Failed to load source data: {e}")
                pbar.update(1)
                continue

            with open(label_path, 'r') as f:
                labels = f.readlines()

            point_cloud = self.depth_to_pointcloud(depth_image)
            transformed_point_cloud = self.transform_pointcloud(point_cloud, pose)
            depth_mask = (depth_image > 0) & (depth_image < max_depth)

            for label in labels:
                label_data = list(map(float, label.split()))
                points = np.array(label_data[1:]).reshape(-1, 2)

                mask = np.zeros(depth_image.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [points.astype(np.int32)], 1)

                seg_mask = depth_mask & (mask > 0)
                seg_indices = np.where(seg_mask)    
                
                seg_points = transformed_point_cloud.points[seg_indices]
                seg_colors = rgb_image[seg_indices]

                rgb_points = transformed_point_cloud.points[depth_mask]
                rgb_colors = rgb_image[depth_mask]

                seg_point_cloud.points.extend(seg_points)
                seg_point_cloud.colors.extend(seg_colors)
                rgb_point_cloud.points.extend(rgb_points)
                rgb_point_cloud.colors.extend(rgb_colors)

            pbar.update(1)
        pbar.close()

        end_time = time.time()
        print(f"Time consumption for merging: {end_time - start_time}")
        o3d.io.write_point_cloud(self.seg_pointcloud_save_path, seg_point_cloud)
        o3d.io.write_point_cloud(self.rgb_pointcloud_save_path, rgb_point_cloud)
        
    def show_contours(self, use_pgm=False):
        if use_pgm:
            image = Image.open(self.map_pgm)
            image = np.array(image)
            with open(self.map_yaml, 'r') as f:
                metadata = yaml.safe_load(f)
            resolution = metadata['resolution']
            origin = metadata['origin']
            height, width = image.shape
            with open(self.object_info_path, 'r') as f:
                object_info = json.load(f)
            fig, ax = plt.subplots() 
            ax.imshow(image, cmap='gray', origin='lower')
            for class_name, contours in object_info.items():
                for contour in contours:
                    contour = np.array(contour)
                    transformed_contour = np.column_stack((
                        (contour[:, 0] - origin[0]) / resolution ,
                        (origin[1] + height * resolution - contour[:, 1]) / resolution
                    ))
                    ax.plot(transformed_contour[:, 0], transformed_contour[:, 1])

                    center_x = np.mean(transformed_contour[:, 0])
                    center_y = np.mean(transformed_contour[:, 1])
                    ax.text(center_x, center_y, class_name, fontsize=12, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            # origin_x, origin_y = origin[0], origin[1]
            # ax.scatter(origin_x, origin_y, s=100, color='red')
            # ax.text(origin_x, origin_y, 'origin', fontsize=12, ha='center', va='center',
            #         bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.5'))
            ax.set_aspect('equal', adjustable='box')
            plt.title('2D Contours on Map')
            plt.xlabel('X (pixels)')
            plt.ylabel('Y (pixels)')
            plt.show()
        else:
            with open(self.object_info_path, 'r') as f:
                object_info = json.load(f)

            fig, ax = plt.subplots()
            for class_name, contours in object_info.items():
                for contour in contours:
                    contour = np.array(contour)
                    ax.plot(contour[:, 0], contour[:, 1])

                    center_x = np.mean(contour[:, 0])
                    center_y = np.mean(contour[:, 1])
                    ax.text(center_x, center_y, class_name, fontsize=12, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            ax.set_aspect('equal', adjustable='box')
            plt.title('2D Contours')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
        
    def show_pointcloud(self, seg=True):
        if seg:
            point_cloud = o3d.io.read_point_cloud(self.seg_pointcloud_save_path)
        else:
            point_cloud = o3d.io.read_point_cloud(self.rgb_pointcloud_save_path)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(point_cloud)
        vis.run()
        vis.destroy_window()
        
            
if __name__ == "__main__":
    data_dir = "/home/nvidia/maps/20240808"
    trans_camera_base = [0.07992725371130696, -0.04192086603435837, 1.3467514828245077]
    quat_camera_base = [-0.4833235028331, 0.48665728519877355, -0.5084449268770554, 0.520621584939664]
    map2base_link_origin = np.array([[-0.347, 0.938, 0.000, 2.229],
                                    [-0.938, -0.347, -0.000, 9.121],
                                    [-0.000, 0.000, 1.000, 0.255],
                                    [0.000, 0.000, 0.000, 1.000]])
    
    # T_basefootprint_camera = np.array([[-0.00205885, -0.02153475, 0.99976598, 0.10160911],
    #                                     [-0.99993756, 0.01102532, -0.00182172, 0.03182674],
    #                                     [-0.01098351, -0.99970731, -0.0215561, 0.56075885],
    #                                     [ 0., 0., 0., 1.]])

    # trans_camera_footprint = [0.08278859935292791, -0.03032243564439939, 1.3482014910932798]
    # quat_camera_footprint = [-0.48836894018639176, 0.48413701319615116, -0.5135400532533373, 0.5132092598729002]
    
    # ssmap = MapBuilder(data_dir, trans_camera_base, quat_camera_base, map2base_link_origin, is_map_origin=False)
    # ssmap = mapbuilder(data_dir, trans_camera_footprint, quat_camera_footprint)
    ssmap = MapBuilder(data_dir, trans_camera_base, quat_camera_base, map2base_link_origin, is_map_origin=True)
    ssmap.data_load()
    ssmap.seg_rgb(confidence=0.5)
    ssmap.build_2dmap(max_depth=6000, DBSCAN_eps=0.1, DBSCAN_min_samples=190, voxel_size=0.05)
    # ssmap.show_contours(use_pgm=False)
    # ssmap.merge_pointcloud()