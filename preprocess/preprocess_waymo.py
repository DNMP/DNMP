import numpy as np
import os
import cv2
import open3d as o3d
from tqdm import tqdm

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils

import argparse
parser = argparse.ArgumentParser(description='waymo process')
parser.add_argument('--root_dir', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')

args = parser.parse_args()

root_dir = args.root_dir
save_dir = args.save_dir

seq_dict = {}
seq_dict['seq_1'] = 'validation_0000/segment-10247954040621004675_2180_000_2200_000_with_camera_labels.tfrecord'
seq_dict['seq_2'] = 'validation_0000/segment-1071392229495085036_1844_790_1864_790_with_camera_labels.tfrecord'
seq_dict['seq_3'] = 'validation_0000/segment-11037651371539287009_77_670_97_670_with_camera_labels.tfrecord'
seq_dict['seq_4'] = 'validation_0001/segment-13469905891836363794_4429_660_4449_660_with_camera_labels.tfrecord'
seq_dict['seq_5'] = 'validation_0002/segment-14333744981238305769_5658_260_5678_260_with_camera_labels.tfrecord'
seq_dict['seq_6'] = 'validation_0002/segment-14663356589561275673_935_195_955_195_with_camera_labels.tfrecord'
range_dict = {}
range_dict['seq_1'] = [0, 80]
range_dict['seq_2'] = [135,197]
range_dict['seq_3'] = [0, 164]
range_dict['seq_4'] = [0, 197]
range_dict['seq_5'] = [0, 198]
range_dict['seq_6'] = [0, 197]

for seq in seq_dict.keys():
    print(f'Processing sequence {seq}')
    data_path = os.path.join(root_dir, seq_dict[seq])
    seq_save_dir = os.path.join(save_dir, seq)
    os.makedirs(seq_save_dir, exist_ok=True)
    datafile = WaymoDataFileReader(data_path)
    start_idx = range_dict[seq][0]
    end_idx = range_dict[seq][1]

    train_save_dir = os.path.join(seq_save_dir, 'train_imgs')
    os.makedirs(train_save_dir, exist_ok=True)
    test_save_dir = os.path.join(seq_save_dir, 'test_imgs')
    os.makedirs(test_save_dir, exist_ok=True)

    c2ws_train = []
    c2ws_test = []

    data_idx = 0

    print("Processing image data...")
    for frameno, frame in tqdm(enumerate(datafile)):
        if frameno >= start_idx and frameno <=end_idx:
            camera_name = dataset_pb2.CameraName.FRONT
            camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
            camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4) # camera to vehicle
            vehicle_pose = np.array(frame.pose.transform).reshape(4,4)
            camera_intrinsic = np.array(camera_calibration.intrinsic)
            fx = camera_intrinsic[0]
            fy = camera_intrinsic[1]
            cx = camera_intrinsic[2]
            cy = camera_intrinsic[3]
            intrinsic = np.array([fx, fy, cx, cy])
            intrinsic_M = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
            opengl2camera = np.array([[0., 0., -1., 0.],
                                [-1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 0., 1.]])
            c2v = np.matmul(camera_extrinsic, opengl2camera)
            c2w_opengl = np.matmul(vehicle_pose, c2v)
            c2w_kitti = c2w_opengl
            c2w_kitti[:,2] = -c2w_kitti[:,2]
            c2w_kitti[:,1] = -c2w_kitti[:,1]
            camera = utils.get(frame.images, camera_name)
            img = utils.decode_image(camera)
            img_w = img.shape[1]
            img_h = img.shape[0]
            if data_idx % 10 == 0 and data_idx != 0:
                cv2.imwrite(os.path.join(test_save_dir, '%06d.png'%frameno), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                c2ws_test.append(c2w_kitti)
            else:
                cv2.imwrite(os.path.join(train_save_dir, '%06d.png'%frameno), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                c2ws_train.append(c2w_kitti)
            data_idx += 1

    c2ws_train = np.stack(c2ws_train, axis=0)
    c2ws_test = np.stack(c2ws_test, axis=0)
    np.save(f'{seq_save_dir}/train_poses.npy', c2ws_train)
    np.save(f'{seq_save_dir}/test_poses.npy', c2ws_test)
    np.save(f'{seq_save_dir}/intrinsics.npy', intrinsic_M)
    center_point = c2ws_train[c2ws_train.shape[0]//2]
    np.save(f'{seq_save_dir}/center_point.npy', center_point)

    datafile = WaymoDataFileReader(data_path)
    pts = []
    print("Processing LiDAR data...")
    for frameno, frame in tqdm(enumerate(datafile)):
        if frameno >= start_idx and frameno <=end_idx:
            for laser_id in range(1, 6):
                laser_name = dataset_pb2.LaserName.Name.DESCRIPTOR.values_by_number[laser_id].name
                laser = utils.get(frame.lasers, laser_id)
                laser_calibration = utils.get(frame.context.laser_calibrations, laser_id)
                ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)
                laser_extrinsic = np.array(laser_calibration.extrinsic.transform).reshape(4,4) # laser to vehicle
                vehicle_pose = np.array(frame.pose.transform).reshape(4,4)
                l2w = vehicle_pose
                pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
                pcl = l2w.dot(np.concatenate([pcl, np.ones((pcl.shape[0], 1))], axis=1).T).T
                pts.append(pcl)

    pts = np.concatenate(pts, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[...,:3])
    
    print("Denosing...")
    down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
    cl, ind = down_pcd.remove_radius_outlier(nb_points=10, radius=0.2)
    np.savez_compressed(f'{seq_save_dir}/pcd.npz', pts=np.asarray(cl.points))