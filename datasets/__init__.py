from .kitti360_dataset import KITTI360Dataset
from .waymo_dataset import WaymoDataset

dataset_dict = {
    'kitti360':KITTI360Dataset,
    'waymo':WaymoDataset
}