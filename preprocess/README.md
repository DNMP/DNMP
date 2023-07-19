# Data pre-process

### KITTI-360
Download [KITTI-360 dataset](https://www.cvlibs.net/datasets/kitti-360) (Perspective Images for Train & Val, Vehicle Poses, Calibrations). 
The downloaded data should be organized as 
```
ROOT_DIR
├── data_2d_raw
│   ├── ...
│   ├── 2013_05_28_drive_0009_sync
│   │   ├── image_00
│   │   │   ├── data_rect
│   │   ├── image_01
│   │   │   ├── data_rect
│   ├── ...
├── calibration
│   ├── calib_cam_to_pose.txt
│   ├── perspective.txt
│   ├── ...
├── data_poses
│   ├── ...
│   ├── 2013_05_28_drive_0009_sync
│   │   ├── cam0_to_world.txt
│   │   ├── poses.txt
│   ├── ...
├── ...
```
Process the data by running: `python preprocess/preprocess_kitti.py --root_dir ROOT_DIR --save_dir SAVE_DIR`, the processed data will be saved to `SAVE_DIR`.

If you want to train the geometry yourself, please install COLMAP according to the [instruction](https://colmap.github.io/install.html) and then run `sh preprocess/run_colmap_kitti.sh`. (Please specify `SEQUENCE`, `PROJECT_ROOT`, `ROOT_DIR` in the script.)

If you want to use Metashape to reconstruct the point cloud, please download the [software](https://www.agisoft.com/) and run `sh preprocess/run_metashape_kitti.sh` to generate cameras.xml for specified sequence. (Please specify `SEQUENCE`, `PROJECT_ROOT`, `ROOT_DIR` in the script.) After loaded the images, you can import `cameras.xml` to use ground truth poses and camera intrinsics for sparse and dense reconstruction.

### Waymo Open Dataset
Download the validation set of [Waymo Open Dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_3_1/validation;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false).
The downloaded data should be organized as 
```
ROOT_DIR
├── validation_0000
│   ├── ...
│   ├── segment-10247954040621004675_2180_000_2200_000_with_camera_labels.tfrecord
│   ├── ...
├── validation_0001
├── validation_0002
├── ...
```
Install [simple_waymo_open_dataset_reader](https://github.com/gdlg/simple-waymo-open-dataset-reader) to read tfrecord data.

Process the data by running `python preprocess/preprocess_waymo.py --root_dir ROOT_DIR --save_dir SAVE_DIR`.
