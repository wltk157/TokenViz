from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms.functional as F
import numpy
import pandas
from tqdm import tqdm


class ClassficationDataset(Dataset):
    def __init__(self, img_path, labels, task='CC', size=[224, 224]):

        self.img_path = img_path
        self.labels = labels
        self.task = task
        self.size = size

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):

        if self.task == 'CD':

            img1_path, img2_path = self.img_path[item]
            img1, img2 = numpy.load(img1_path), numpy.load(img2_path)
            img1, img2 = self.transform(img1), self.transform(img2)
            img = [img1, img2]
        else:
            img = numpy.load(self.img_path[item])
            img = self.transform(img)

        return img, self.labels[item]

    def transform(self, img):

        img = F.resize(torch.Tensor(img), (self.size[0] + 6, self.size[1] + 6))

        img = F.center_crop(img, (self.size[0], self.size[1]))

        img = img.float()

        channel = img.size(dim=0)

        mean = torch.zeros(channel)
        std = torch.ones(channel)
        img = (img - mean.view(channel, 1, 1)) / std.view(channel, 1, 1)

        return img


def get_dataloader(task, IR_path, pairs_file, batch_size, num_workers, shuffle=True, size=[224, 224]):
    if task != 'CD' and task != 'CC' and task != 'DD':
        print("Warning: incorrect task type!")

    def create_label_mapping(datasets):
        labels = set()
        for _, row in datasets.iterrows():
            labels.add(row['func'])
        mapping = {}
        indx = 0
        for x in labels:
            mapping[x] = indx
            indx += 1
        return mapping

    pairs = pandas.read_csv(pairs_file)

    train_pair = []
    train_label = []

    valid_pair = []
    valid_label = []
    if task != 'DD':
        test_pair = []
        test_label = []

    print("Processing...")

    labelmapping = create_label_mapping(pairs)

    for index, row in tqdm(pairs.iterrows()):
        if task != 'CD':

            img_path = IR_path + str(row['id']) + ".npy"
            raw_label = row['func']
            label = labelmapping[raw_label] if raw_label in labelmapping else 0
        else:

            img1_path = IR_path + str(row['id1']) + ".npy"
            img2_path = IR_path + str(row['id2']) + ".npy"
            img_path = [img1_path, img2_path]
            label = 0 if row['type'] <= 0 else 1

        if row['divide_type'] == 0:
            train_pair.append(img_path)
            train_label.append(label)
        elif row['divide_type'] == 1:
            valid_pair.append(img_path)
            valid_label.append(label)
        elif task != 'DD':
            test_pair.append(img_path)
            test_label.append(label)

    print("Building dataset...")
    train_dataset = ClassficationDataset(train_pair, train_label, task, size)

    valid_dataset = ClassficationDataset(valid_pair, valid_label, task, size)
    if task != 'DD':
        test_dataset = ClassficationDataset(test_pair, test_label, task, size)

    print("Building dataloader...")
    train_dataloader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    valid_dataloader = DataLoader(valid_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    if task != 'DD':
        test_dataloader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True)
        return train_dataloader, valid_dataloader, test_dataloader
    else:
        return train_dataloader, valid_dataloader
