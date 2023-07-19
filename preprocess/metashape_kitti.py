import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import xml.dom.minidom as xmldom

parser = argparse.ArgumentParser(description='metashape_kitti')
parser.add_argument('--project_path', type=str, default='')
parser.add_argument('--image_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')

DRIVE = '2013_05_28_drive_0009_sync'

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path
    image_path = args.image_path
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
    
    cx = P_rect_00[0, 2]
    cy = P_rect_00[1, 2]
    fx = P_rect_00[0, 0]
    fy = P_rect_00[1, 1]
    img_h = 376
    img_w = 1408
    cx_ = cx - img_w / 2
    cy_ = cy - img_h / 2
    img_fns = sorted(os.listdir(image_path))

    doc = xmldom.Document()
    root_node = doc.createElement('document')
    root_node.setAttribute('version', '1.5.0')
    doc.appendChild(root_node)
    chunk_node = doc.createElement('chunk')
    chunk_node.setAttribute('label', 'Chunk 1')
    chunk_node.setAttribute('enabled', 'true')
    root_node.appendChild(chunk_node)
    sensors_node = doc.createElement('sensors')
    sensors_node.setAttribute('next_id', '1')
    chunk_node.appendChild(sensors_node)
    sensor_node = doc.createElement('sensor')
    sensor_node.setAttribute('id', '0')
    sensor_node.setAttribute('label', 'unknown')
    sensor_node.setAttribute('type', 'frame')
    sensors_node.appendChild(sensor_node)

    resolution_node = doc.createElement('resolution')
    resolution_node.setAttribute('width', '1408')
    resolution_node.setAttribute('height', '376')

    sensor_node.appendChild(resolution_node)
    property_node = doc.createElement('property')
    property_node.setAttribute('name', 'fixed')
    property_node.setAttribute('value', 'true')
    sensor_node.appendChild(property_node)
    property_node = doc.createElement('property')
    property_node.setAttribute('name', 'layer_index')
    property_node.setAttribute('value', '0')
    sensor_node.appendChild(property_node)

    bands_node = doc.createElement('bands')
    band_node = doc.createElement('band')
    band_node.setAttribute('label', 'Red')
    bands_node.appendChild(band_node)
    band_node = doc.createElement('band')
    band_node.setAttribute('label', 'Green')
    bands_node.appendChild(band_node)
    band_node = doc.createElement('band')
    band_node.setAttribute('label', 'Blue')
    bands_node.appendChild(band_node)
    sensor_node.appendChild(bands_node)

    data_type_node = doc.createElement('data_type')
    data_type_value = doc.createTextNode('uint8')
    data_type_node.appendChild(data_type_value)
    sensor_node.appendChild(data_type_node)

    calibration_node = doc.createElement('calibration')
    calibration_node.setAttribute('type', 'frame')
    calibration_node.setAttribute('class', 'initial')
    sensor_node.appendChild(calibration_node)
    resolution_node = doc.createElement('resolution')
    resolution_node.setAttribute('width', '1408')
    resolution_node.setAttribute('height', '376')
    calibration_node.appendChild(resolution_node)
    f_node = doc.createElement('f')
    f_value = doc.createTextNode('552.554261')
    f_node.appendChild(f_value)
    calibration_node.appendChild(f_node)
    cx_node = doc.createElement('cx')
    cx_value = doc.createTextNode(str(cx_))
    cx_node.appendChild(cx_value)
    calibration_node.appendChild(cx_node)
    cy_node = doc.createElement('cy')
    cy_value = doc.createTextNode(str(cy_))
    cy_node.appendChild(cy_value)
    calibration_node.appendChild(cy_node)

    black_level_node = doc.createElement('black_level')
    black_level_value = doc.createTextNode('0 0 0')
    black_level_node.appendChild(black_level_value)
    sensor_node.appendChild(black_level_node)

    sensitivity_node = doc.createElement('sensitivity')
    sensitivity_value = doc.createTextNode('1 1 1')
    sensitivity_node.appendChild(sensitivity_value)
    sensor_node.appendChild(sensitivity_node)

    cameras_node = doc.createElement('cameras')
    cameras_node.setAttribute('next_id', str(len(img_fns)))
    cameras_node.setAttribute('next_group_id', '1')

    for i in range(len(img_fns)):
        camera_node = doc.createElement('camera')
        camera_node.setAttribute('id', str(i))
        camera_node.setAttribute('sensor_id', '0')
        img_name = img_fns[i]
        img_id = img_name.split('.')[0]
        camera_node.setAttribute('label', img_id)
        one_pose = c2w_dict[img_name].flatten()
        one_pose_string = ' '.join(map(str, one_pose))
        transform_node = doc.createElement('transform')
        transform_value = doc.createTextNode(one_pose_string)
        transform_node.appendChild(transform_value)
        camera_node.appendChild(transform_node)

        rotation_covariance_node = doc.createElement('rotation_covariance')
        rotation_covariance_value = doc.createTextNode('0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0')
        rotation_covariance_node.appendChild(rotation_covariance_value)
        camera_node.appendChild(rotation_covariance_node)

        location_covariance_node = doc.createElement('location_covariance')
        location_covariance_value = doc.createTextNode('0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0')
        location_covariance_node.appendChild(location_covariance_value)
        camera_node.appendChild(location_covariance_node)

        cameras_node.appendChild(camera_node)

    chunk_node.appendChild(cameras_node)

    with open(f'{project_path}/cameras.xml', "w", encoding="utf-8") as f:
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")