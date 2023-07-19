import sqlite3
import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description='colmap_kitti')
parser.add_argument('--project_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')

DRIVE = '2013_05_28_drive_0009_sync'

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path
    project_path = args.project_path

    pose_fn = f'{data_path}/data_poses/{DRIVE}/poses.txt'
    intrinsic_fn = f'{data_path}/calibration/perspective.txt'
    cam2pose_fn = f'{data_path}/calibration/calib_cam_to_pose.txt'
    poses = np.loadtxt(pose_fn)
    img_id = poses[:, 0].astype(np.int32)
    poses = poses[:, 1:].reshape(-1, 3, 4)
    pose_dict = {}
    for i in range(len(img_id)):
        img_name = f'{img_id[i]:010d}.png'
        pose_dict[img_name] = poses[i]

    def load_intrinsics(intrinsics_fn):
        with open(intrinsics_fn, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                line = line.split(' ')
                if line[0] == 'P_rect_00:':
                    P_rect_00 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
                elif line[0] == 'P_rect_01:':
                    P_rect_01 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
                elif line[0] == 'R_rect_00:':
                    R_rect_00 = np.array(line[1:], dtype=np.float32).reshape(3, 3)
                elif line[0] == 'R_rect_01:':
                    R_rect_01 = np.array(line[1:], dtype=np.float32).reshape(3, 3)
        return P_rect_00, P_rect_01, R_rect_00, R_rect_01

    def load_cam_to_pose(cam_to_pose_fn):
        with open(cam_to_pose_fn, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                line = line.split(' ')
                if line[0] == 'image_00:':
                    c2p_00 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
                elif line[0] == 'image_01:':
                    c2p_01 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
        return c2p_00, c2p_01

    P_rect_00, P_rect_01, R_rect_00_, R_rect_01_ = load_intrinsics(intrinsic_fn)
    c2p_00, c2p_01 = load_cam_to_pose(cam2pose_fn)
    c2p_00 = np.concatenate([c2p_00, np.array([[0, 0, 0, 1]])], axis=0)
    c2p_01 = np.concatenate([c2p_01, np.array([[0, 0, 0, 1]])], axis=0)
    R_rect_00 = np.eye(4)
    R_rect_00[:3, :3] = R_rect_00_
    R_rect_01 = np.eye(4)
    R_rect_01[:3, :3] = R_rect_01_
    c2w_dict = {}
    for img_name in pose_dict.keys():
        pose = pose_dict[img_name]
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
        c2w_00 = np.matmul(np.matmul(pose, c2p_00), np.linalg.inv(R_rect_00))
        c2w_01 = np.matmul(np.matmul(pose, c2p_01), np.linalg.inv(R_rect_01))
        c2w_dict[f'00_{img_name}'] = c2w_00
        c2w_dict[f'01_{img_name}'] = c2w_01

    db_fn = f'{project_path}/database.db'
    conn = sqlite3.connect(db_fn)
    c = conn.cursor()
    c.execute('SELECT * FROM images')
    result = c.fetchall()

    out_fn = f'{project_path}/id_names.txt'
    with open(out_fn, 'w') as f:
        for i in result:
            f.write(str(i[0]) + ' ' + i[1] + '\n')
    f.close()

    path_idname = f'{project_path}/id_names.txt'
    
    f_id_name= open(path_idname, 'r')
    f_id_name_lines= f_id_name.readlines()

    images_save_dir = f'{project_path}/created/sparse/model'
    if not os.path.exists(images_save_dir):
        os.makedirs(images_save_dir)

    f_w= open(f'{project_path}/created/sparse/model/images.txt', 'w')
    id_names = []
    for l in f_id_name_lines:
        l = l.strip().split(' ')
        id_ = int(l[0])
        name = l[1]
        id_names.append([id_, name])

    for i in range(len(id_names)):
        id_ = id_names[i][0]
        name = id_names[i][1]
        
        transform = c2w_dict[name]
        transform = np.linalg.inv(transform)

        r = R.from_matrix(transform[:3,:3])
        rquat= r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
        rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
        out = np.concatenate((rquat, transform[:3, 3]), axis=0)
        id_ = id_names[i][0]
        name = id_names[i][1]
        f_w.write(f'{id_} ')
        f_w.write(' '.join([str(a) for a in out.tolist()] ) )
        f_w.write(f' 1 {name}')
        f_w.write('\n\n')
    
    f_w.close()

    cameras_fn = os.path.join(images_save_dir, 'cameras.txt')
    fx = P_rect_00[0, 0]
    fy = P_rect_00[1, 1]
    cx = P_rect_00[0, 2]
    cy = P_rect_00[1, 2]
    with open(cameras_fn, 'w') as f:
        f.write(f'1 SIMPLE_PINHOLE 1408 376 {fx} {cx} {cy}')

    points3D_fn = os.path.join(images_save_dir, 'points3D.txt')
    os.system(f'touch {points3D_fn}')