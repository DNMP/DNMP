from core.config import get_config
from core.tester import Tester
from core.train_utils import setup_seed

if __name__ == '__main__':

    config = get_config()

    setup_seed(0)
    tester = Tester(config)
    
    tester.test()