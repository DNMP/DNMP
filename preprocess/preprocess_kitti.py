import os
import numpy as np
import shutil

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='kitti process')
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    args = parser.parse_args()

    DRIVE = '2013_05_28_drive_0009_sync'
    root_dir = args.root_dir
    save_dir = args.save_dir

    shutil.copytree(os.path.join(root_dir, 'calibration'), os.path.join(save_dir, 'calibration'))
    shutil.copytree(os.path.join(root_dir, 'data_poses'), os.path.join(save_dir, 'data_poses'))

    data_dir = f'{root_dir}/data_2d_raw/{DRIVE}'

    image_00_dir = os.path.join(data_dir, 'image_00/data_rect')
    image_01_dir = os.path.join(data_dir, 'image_01/data_rect')
    img_00_fns = sorted(os.listdir(image_00_dir))
    img_01_fns = sorted(os.listdir(image_01_dir))

    start_end_list = [[715,795],[880,960],[1102,1182],[2170,2250],[2900,2980]]

    for idx in range(len(start_end_list)):
        seq_name = f'seq_{idx+1}'
        print(f'Processing sequence: {seq_name}.')
        start = start_end_list[idx][0]
        end = start_end_list[idx][1]
        seq_save_dir = os.path.join(save_dir, seq_name)
        seq_save_dir_00 = os.path.join(seq_save_dir, 'image_00')
        seq_save_dir_01 = os.path.join(seq_save_dir, 'image_01')
        os.makedirs(seq_save_dir_00, exist_ok=True)
        os.makedirs(seq_save_dir_01, exist_ok=True)
        for i in range(start, end):
            img_00_fn = img_00_fns[i]
            img_01_fn = img_01_fns[i]
            img_00_src = os.path.join(image_00_dir, img_00_fn)
            img_01_src = os.path.join(image_01_dir, img_01_fn)
            img_00_dst = os.path.join(seq_save_dir_00, img_00_fn)
            img_01_dst = os.path.join(seq_save_dir_01, img_01_fn)
            shutil.copy(img_00_src, img_00_dst)
            shutil.copy(img_01_src, img_01_dst)
        
        train_save_dir = os.path.join(seq_save_dir, 'train_imgs')
        test_save_dir = os.path.join(seq_save_dir, 'test_imgs')

        os.makedirs(train_save_dir, exist_ok=True)
        os.makedirs(test_save_dir, exist_ok=True)

        img_fns = os.listdir(seq_save_dir_00)
        for img_fn in img_fns:
            img_id = int(img_fn.split('.')[0])
            if img_id % 2 == 0:
                shutil.copy(os.path.join(seq_save_dir_00, img_fn), os.path.join(train_save_dir, f'00_{img_fn}'))
                shutil.copy(os.path.join(seq_save_dir_01, img_fn), os.path.join(train_save_dir, f'01_{img_fn}'))
            else:
                shutil.copy(os.path.join(seq_save_dir_00, img_fn), os.path.join(test_save_dir, f'00_{img_fn}'))

        img_names = sorted(os.listdir(train_save_dir))
        center_img_name = img_names[len(img_names) // 4]

        poses_fn = f'{root_dir}/data_poses/{DRIVE}/poses.txt'
        intrinsic_fn = f'{root_dir}/calibration/perspective.txt'
        cam2pose_fn = f'{root_dir}/calibration/calib_cam_to_pose.txt'
        poses = np.loadtxt(poses_fn)
        img_id = poses[:, 0].astype(np.int32)
        poses = poses[:, 1:].reshape(-1, 3, 4)
        pose_dict = {}
        for i in range(len(img_id)):
            img_name = f'{img_id[i]:010d}.png'
            pose_dict[img_name] = poses[i]
        
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
        
        center_point = c2w_dict[center_img_name]

        np.save(f'{seq_save_dir}/center_point.npy', center_point)