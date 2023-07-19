import logging
import coloredlogs
import os
import torch
import numpy as np
import random

def get_logger(log_path):

    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info('Output and logs will be saved to {}'.format(log_path))

    return logger

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        if isinstance(val, np.ndarray):
            n = val.size
            val = val.mean()
        elif isinstance(val, torch.Tensor):
            n = val.nelement()
            val = val.mean().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val**2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2

def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True