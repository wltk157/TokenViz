import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from train import train
from test import test
from visualize import *
import time


def block_experiment():
    config = {
        "use_gpu": True,
        "gpu_id": '0',
        "num_epoch": 100,
        "init_epoch": 0,
        "optimizer_type": "adam",
        "input_shape": [1, 224, 224],
        "IR_path": "IR_block/",
        "dataset_file": "dataset/pairs.csv",
        "fragments_file": "dataset/fragments.csv",
        "init_lr": 1e-2,
        "min_lr": 1e-3,
        "momentum": 0.9,
        "lr_decay_type": 'cos',
        "save_period": 10,
        "save_dir": "logs_block",
        "batch_size": 64,
        "num_workers": 10,
        "weight_decay": 5e-4,
        "net": "resnet",
        "class_num": 2,
        "num_token": [80, 80],
        "block_size": [4, 4],
        "variable": True,
        "newline": False,
        "dropout": False
    }

    print(time.time())
    if not os.path.exists(config['IR_path']):
        print("IR not found, creating visual representation...")
        block_visualize(config['fragments_file'], config['IR_path'], config['num_token'][0],
                        config['num_token'][1], config['block_size'][0], config['block_size'][1],
                        config['variable'], config['newline'])
    else:
        print("IR found, start training...")
    print(time.time())
    train(config)
    test(config)


if __name__ == "__main__":
    block_experiment()
