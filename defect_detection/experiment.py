import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from train import learn_on_a_dataset
from defect_detection.expreport import report
from visualize import *


def block_experiment():
    config = {
        "use_gpu": True,
        "gpu_id": '1',
        "num_epoch": 100,
        "init_epoch": 0,
        "optimizer_type": "adam",
        "input_shape": [1, 224, 224],
        "dataset_path": "dataset",
        "IR_path": "IR_block/",
        "init_lr": 1e-2,
        "min_lr": 1e-3,
        "momentum": 0.9,
        "lr_decay_type": 'cos',
        "save_period": 10,
        "save_dir": "logs_block-4",
        "batch_size": 64,
        "num_workers": 10,
        "weight_decay": 5e-4,
        "net": "resnet50",
        "class_num": 2,
        "run_times": 10,
        "num_token": [80, 80],
        "block_size": [4, 4],
        "variable": True,
        "newline": False,
        "w2v_model_name": "word2vec.pth",
        'dropout': True
    }

    dataset_path = config['dataset_path']
    IR_path = config['IR_path']
    run_times = config['run_times']
    save_dir = config['save_dir']
    w2v_model_name = config['w2v_model_name']

    dirs = os.listdir(dataset_path, )

    block_visualize(dataset_path, IR_path, w2v_model_name, config['num_token'][0],
                    config['num_token'][1], config['block_size'][0], config['block_size'][1],
                    config['variable'], config['newline'])

    start_time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime())
    report.set_save_file("block_experiment_%s.csv" % start_time_str)

    for i in range(run_times):

        report.run = i
        print("Run: %s" % i)

        for dir in dirs:
            print("Training:" + dir)

            config['dataset_path'] = dataset_path + "/" + dir + "/fragments.csv"
            config['IR_path'] = dataset_path + "/" + dir + "/" + IR_path
            config['save_dir'] = save_dir + "/" + str(i) + "/" + dir
            os.makedirs(config['save_dir'], exist_ok=True)

            report.dataset = dir

            learn_on_a_dataset(config)


if __name__ == "__main__":
    block_experiment()
