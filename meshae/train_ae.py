import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='MeshAutoEncoder')

parser.add_argument('--mesh_dir', type=str, default=None)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_step', type=int, default=10000)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--checkpoint_dir', type=str, default=None)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--max_iter', type=int, default=200000)

def get_config():
    args = parser.parse_args()
    return args

from meshae.trainer import Trainer
from core.train_utils import setup_seed

if __name__ == '__main__':

    config = get_config()

    setup_seed(0)
    trainer = Trainer(config)

    trainer.train()