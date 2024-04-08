import os
import sys

sys.path.append("..")
from shared.learn import count_parameters
from shared.utils.dataloader import get_dataloader
from tqdm import tqdm
import torch
import torch.nn as nn
from nets.resnet import ResNet50

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def test_model(model, loss, test_iter, save_dir, device, epoch_step_test):
    try:
        test_loss = 0
        test_total_accuracy = 0

        n = 0
        test_p = 0
        test_r = 0
        test_f = 0

        print('Start Test')

        pbar = tqdm(total=epoch_step_test, desc=f'Test', postfix=dict, mininterval=0.3)

        model.eval()
        predict_all = []
        labels_all = []

        for iteration, batch in enumerate(test_iter):
            images, labels = batch[0], batch[1]
            images = images.permute(0, 1, 2, 3).float()

            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                output = loss(outputs, labels)

                _, predicted = torch.max(outputs, dim=1)
                equal = torch.eq(predicted, labels)
                accuracy = torch.mean(equal.float())

                prep = predicted.cpu().numpy()

                label = labels.cpu().numpy()

                predict_all.extend(prep)
                labels_all.extend(label)
                p, r, f, _ = precision_recall_fscore_support(label, prep, average='macro', zero_division=1)

            test_loss += output.item()
            test_total_accuracy += accuracy.item()

            test_p += p
            test_r += r
            test_f += f
            n = n + 1

            pbar.set_postfix(**{'test_loss': test_loss / (iteration + 1),
                                'acc': test_total_accuracy / (iteration + 1),
                                'precision': test_p / n,
                                'recall': test_r / n,
                                "f1-score": test_f / n})
            pbar.update(1)
        pbar.close()

        accuracy = accuracy_score(labels_all, predict_all)
        precision, recall, f1, _ = precision_recall_fscore_support(labels_all, predict_all, average='macro',
                                                                   zero_division=1)

        with open(os.path.join(save_dir, "test_result.csv"), 'w+') as f:
            f.write("Accuracy,Precision,Recall,F1,Loss\n")
            f.write("%.5f,%.5f,%.5f,%.5f,%.5f" % (accuracy, precision, recall, f1, test_loss / len(test_iter)))
        print('Finish test')

    except:
        print('\033[0;31mAbnormal exit, test terminated!!!\033[0m')
        print('\033[0;31mModel has been saved!!!\033[0m')


def test(config):
    use_gpu = config['use_gpu']
    gpu_id = config['gpu_id']
    batch_size = config['batch_size']
    input_shape = config['input_shape']
    IR_path = config['IR_path']
    dataset_file = config['dataset_file']
    save_dir = config['save_dir']
    num_workers = config['num_workers']
    network = config['net']
    class_num = config['class_num']

    with open(os.path.join(save_dir, "log_test.txt"), 'a') as f:

        f.write("----------------\n")
        for key, value in config.items():
            f.write(key + ": " + str(value))
            f.write("\n")
        f.write("----------------\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Init model...")
    model = ResNet50(input_shape[0], class_num)
    num_parameters = count_parameters(model)

    print(f"Num parameters:{num_parameters}")

    loss = nn.CrossEntropyLoss()

    if config['use_gpu']:
        model.cuda()

    print("Getting dataloader...")
    _, _, test_iter = get_dataloader('CC', IR_path, dataset_file, batch_size, num_workers,
                                     size=[input_shape[1], input_shape[2]])

    num_test = len(test_iter) * batch_size

    epoch_step_test = num_test // batch_size

    if epoch_step_test == 0:
        raise ValueError("The dataset is too small, please expand the data set.")

    if os.path.exists(os.path.join(save_dir, "model.pth")):
        model_param = torch.load(os.path.join(save_dir, "model.pth"))
        model_p = model_param['model']
        model.load_state_dict(model_p)
        test_model(model, loss, test_iter, save_dir, device, epoch_step_test)
    else:
        print("Model not found, please train the model first")
