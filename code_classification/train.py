import os
import sys

sys.path.append("..")
from shared.learn import count_parameters
from shared.utils.dataloader import get_dataloader
from shared.utils.lr_utils import get_lr, set_optimizer_lr, get_lr_scheduler
from tqdm import tqdm
from shared.utils.loggerSaver import logger_saver
import traceback
import time

import torch
import torch.nn as nn
from nets.resnet import ResNet50

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def train_model(model, num_epoch, train_iter, valid_iter, loss, optimizer, lr_adjust_function, save_dir, save_period,
                device, epoch_step, epoch_step_val, init_epoch=0):
    metrics = {
        'acc': 0,
        'P': 0,
        'R': 0,
        'F1': 0
    }

    if init_epoch >= num_epoch:
        print("Training is complete!")
    try:
        for current_epoch in range(init_epoch, num_epoch):
            set_optimizer_lr(optimizer, lr_adjust_function, current_epoch)

            total_loss = 0
            total_accuracy = 0

            val_loss = 0
            val_total_accuracy = 0

            n = 0
            val_p = 0
            val_r = 0
            val_f = 0

            pbar = tqdm(total=epoch_step, desc=f'train:Epoch {current_epoch + 1}/{num_epoch}', postfix=dict,
                        mininterval=0.3)
            model.train()
            t1 = time.time()

            for iteration, batch in enumerate(train_iter):
                images, labels = batch[0], batch[1]
                images = images.permute(0, 1, 2, 3).float()

                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                output = loss(outputs, labels)

                output.backward()
                optimizer.step()

                with torch.no_grad():
                    _, max_indices = torch.max(nn.Sigmoid()(outputs), dim=1)
                    equal = torch.eq(max_indices, labels)
                    accuracy = torch.mean(equal.float())

                total_loss += output.item()
                total_accuracy += accuracy.item()

                pbar.set_postfix(**{'loss': total_loss / (iteration + 1),
                                    'acc': total_accuracy / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
            pbar.close()
            t2 = time.time()
            train_time = t2 - t1

            pbar2 = tqdm(total=epoch_step_val, desc=f'valid:Epoch {current_epoch + 1}/{num_epoch}', postfix=dict,
                         mininterval=0.3)

            model.eval()
            predict_all = []
            labels_all = []
            t1 = time.time()

            for iteration, batch in enumerate(valid_iter):
                images, labels = batch[0], batch[1]
                images = images.permute(0, 1, 2, 3).float()

                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(images)
                    output = loss(outputs, labels)

                    _, max_indices = torch.max(nn.Sigmoid()(outputs), dim=1)
                    equal = torch.eq(max_indices, labels)
                    accuracy = torch.mean(equal.float())

                    _, max_indices = torch.max(nn.Sigmoid()(outputs), dim=1)
                    prep = max_indices.cpu().numpy()

                    label = labels.cpu().numpy()

                    predict_all.extend(prep)
                    labels_all.extend(label)

                    p, r, f, _ = precision_recall_fscore_support(label, prep, average='macro', zero_division=1)

                val_loss += output.item()
                val_total_accuracy += accuracy.item()

                val_p += p
                val_r += r
                val_f += f
                n = n + 1

                pbar2.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                     'acc': val_total_accuracy / (iteration + 1),
                                     'precision': val_p / n,
                                     'recall': val_r / n,
                                     "f1-score": val_f / n})
                pbar2.update(1)
            pbar2.close()
            t2 = time.time()
            valid_time = t2 - t1

            accuracy = accuracy_score(labels_all, predict_all)
            precision, recall, f1, _ = precision_recall_fscore_support(labels_all, predict_all, average='macro',
                                                                       zero_division=1)

            metrics['acc'] = accuracy
            metrics['P'] = precision
            metrics['R'] = recall
            metrics['F1'] = f1

            print('Finish Validation')
            print('Epoch:' + str(current_epoch + 1) + '/' + str(num_epoch))
            print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
            print('Accuracy: %.5f,Precision: %.5f, Recall: %.5f, F1: %.5f' % (accuracy, precision, recall, f1))

            logger_saver.add_result(total_accuracy / len(train_iter), total_loss / len(train_iter), accuracy,
                                    precision, recall, f1, val_loss / (iteration + 1), train_time, valid_time)
            logger_saver.save(save_dir, model, metrics)
            logger_saver.current_epoch = current_epoch + 1

        print('Finish')
    except:
        print('\033[0;31mAbnormal exit, training terminated!!!\033[0m')
        print('\033[0;31mModel has been saved!!!\033[0m')
        traceback.print_exc()


def train(config):
    use_gpu = config['use_gpu']
    gpu_id = config['gpu_id']
    batch_size = config['batch_size']
    optimizer_type = config['optimizer_type']
    input_shape = config['input_shape']
    epoch = config['num_epoch']
    init_epoch = config['init_epoch']
    IR_path = config['IR_path']
    dataset_file = config['dataset_file']
    init_lr = config['init_lr']
    min_lr = config['min_lr']
    lr_decay_type = config['lr_decay_type']
    save_period = config['save_period']
    save_dir = config['save_dir']
    momentum = config['momentum']
    num_workers = config['num_workers']
    weight_decay = config["weight_decay"]
    network = config['net']
    class_num = config['class_num']
    dropout = config['dropout']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    logger_saver.set_save_file(save_dir + "/log.csv")

    with open(os.path.join(save_dir, "log.txt"), 'a') as f:

        f.write("----------------\n")
        for key, value in config.items():
            f.write(key + ": " + str(value))
            f.write("\n")
        f.write("----------------\n")

    print("Init model...")
    model = ResNet50(input_shape[0], class_num, dropout)
    num_parameters = count_parameters(model)

    print(f"Num parameters:{num_parameters}")

    loss = nn.CrossEntropyLoss()

    model.train()

    if config['use_gpu']:
        model.cuda()

    print("Getting dataloader...")
    train_iter, valid_iter, test_iter = get_dataloader('CC', IR_path,
                                                       dataset_file, batch_size, num_workers,
                                                       size=[input_shape[1], input_shape[2]])

    num_train = len(train_iter) * batch_size
    num_val = len(valid_iter) * batch_size

    wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
    total_step = num_train // batch_size * epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // batch_size) + 1
        print(
            "\n\033[0;31m[Warning] When using %s optimizer, it is recommended to set the total training of the training above %d.\033[0m" % (
                optimizer_type, wanted_step))
        print(
            "\033[0;31m[Warning] The total training data of this operation is %d, and the batch_size is %d. A total of %d EPOCH is trained to calculate the length of the total training step to %d.\033[0m" % (
                num_train, batch_size, epoch, total_step))
        print(
            "\033[0;31m[Warning] Since the total training step is %d，less than the total step length of the proposal %d，it is recommended to set up the total generation to %d。\033[0m" % (
                total_step, wanted_step, wanted_epoch))

    nbs = batch_size
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': torch.optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': torch.optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                               weight_decay=weight_decay)
    }[optimizer_type]

    lr_adjust_function = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epoch)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("The dataset is too small, please expand the data set.")

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    if os.path.exists(os.path.join(save_dir, "model.pth")):

        print("Saved model fouded, restoring training state...")
        print("If that is not what you want, please keep the logs directory empty.")

        model_param = torch.load(os.path.join(save_dir, "model.pth"))
        model_p = model_param['model']
        init_epoch = model_param['epoch']
        model.load_state_dict(model_p)
        print("Initial epoch:%d" % (init_epoch))
        logger_saver.best_model_name = model_param['best_model_name']
        logger_saver.current_epoch = init_epoch
        model.load_state_dict(model_p)
        train_model(model, epoch, train_iter, valid_iter, loss, optimizer, lr_adjust_function, save_dir, save_period,
                    device, epoch_step, epoch_step_val, init_epoch=init_epoch)


    else:

        model.apply(init_weights)
        logger_saver.current_epoch = 0
        train_model(model, epoch, train_iter, valid_iter, loss, optimizer, lr_adjust_function, save_dir, save_period,
                    device, epoch_step, epoch_step_val, init_epoch)
