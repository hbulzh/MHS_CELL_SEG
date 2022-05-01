# -*- coding: utf-8 -*-
"""
Some utility class functions..
Since the input for each model is different, the input features are initialized in each model file.
"""
import math

import PIL
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# main

# 0-1 normalization
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# Creating the coefficient matrix X
def create_x(size, rank):
    x = []
    for i in range(int(2 * size + 1)):
        m = i - size
        row = [m ** j for j in range(rank)]
        x.append(row)
    x = np.mat(x)
    return x


# PCA
def pca(X, numComponents=30, copy=True):
    pca = PCA(n_components=numComponents, copy=copy)
    h, w, c = X.shape[0], X.shape[1], X.shape[2]
    X = X.reshape(h * w, c)
    newX = pca.fit_transform(X)
    newX = newX.reshape(h, w, numComponents)

    return newX


def CalAccuracy(true_label, pred_label, class_num):
    M = 0
    C = np.zeros((class_num + 1, class_num + 1))
    c1 = confusion_matrix(true_label.reshape(-1), pred_label.reshape(-1))
    C[0:class_num, 0:class_num] = c1
    C[0:class_num, class_num] = np.sum(c1, axis=1)
    C[class_num, 0:class_num] = np.sum(c1, axis=0)
    N = np.sum(np.sum(c1, axis=1))
    C[class_num, class_num] = N  # all of the pixel number

    OA = np.trace(C[0:class_num, 0:class_num]) / N
    every_class = np.zeros((class_num + 3,))
    for i in range(class_num):
        acc = C[i, i] / C[i, class_num]
        M = M + C[class_num, i] * C[i, class_num]
        every_class[i] = "{:.2f}".format(acc)

    kappa = (N * np.trace(C[0:class_num, 0:class_num]) - M) / (N * N - M)
    AA = np.sum(every_class, axis=0) / class_num
    every_class[class_num] = "{:.2f}".format(OA)
    every_class[class_num + 1] = "{:.2f}".format(AA)
    every_class[class_num + 2] = "{:.2f}".format(kappa)
    return every_class, C


# 定义迭代器 数据格式(批大小， 序列长度， 输入通道， 长，宽)
def iter(data, label, batch_size):
    with open("./para.json", 'r') as f:
        para_dic = json.load(f)

    length = data.shape[0]  # (8438, 9, 9, 30, 1)
    data = data.reshape(length, para_dic["window_size"], para_dic["window_size"],
                        para_dic["input_dim"], para_dic["short_seq"])
    data = np.swapaxes(data, 1, 4)
    data = np.swapaxes(data, 2, 3)  # (8438, 6, 5, 9, 9)
    num_batches = math.ceil(length / batch_size)
    for index in range(num_batches-1):
        start = index * batch_size
        end = min((index + 1) * batch_size, length)
        yield torch.tensor(data[start:end]), torch.tensor(label[start:end]).long()


def adjust_learning_rate(optimizer, epoch, args):
    lr = args * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def show_img(y, y_, windows_size):
    plt.figure()
    for idx in range(5):
        img_y = y[idx+5].reshape(windows_size, windows_size).astype(int)
        img_y_ = y_[idx+5].reshape(windows_size, windows_size).astype(int)

        for i in range(img_y.shape[0]):
            for j in range(img_y.shape[1]):
                if img_y[i][j] != 0:
                    img_y[i][j] = 255
                if img_y_[i][j] != 0:
                    img_y_[i][j] = 255

        plt.subplot(2, 5, idx + 1)
        img = PIL.Image.fromarray(img_y)
        plt.imshow(img)

        plt.subplot(2, 5, idx + 6)
        img2 = PIL.Image.fromarray(img_y_)
        plt.imshow(img2)

    plt.show()


# data
class genDataSet(Dataset):
    def __init__(self):
        data = np.load('./dataset/data.npy')[:140]
        label = np.load('./dataset/label.npy')[:140]
        Xtrain, Xtest, ytrain, ytest = train_test_split(data, label, test_size=0.1, shuffle=True, random_state=70)
        self._data = Xtrain  # 126
        self._label = ytrain

    def __getitem__(self,idx):
        # Xtrain, Xtest, ytrain, ytest = train_test_split(self._data, self._label, test_size=0.1, shuffle=True, random_state=70)
        img = self._data[idx]
        label = self._label[idx]
        return img, label

    def __len__(self):
        return len(self._data)


class genValidationSet(Dataset):
    def __init__(self):
        data = np.load('./dataset/data.npy')[:140]
        label = np.load('./dataset/label.npy')[:140]
        Xt, X, yt, y = train_test_split(data, label, test_size=0.1, shuffle=True, random_state=70)
        # Xv, Xs, yv, ys = train_test_split(X, y, test_size=0.6, shuffle=True, random_state=70)
        self._data = X
        self._label = y

    def __getitem__(self,idx):
        # Xtrain, Xtest, ytrain, ytest = train_test_split(self._data, self._label, test_size=0.1, shuffle=True, random_state=70)
        img = self._data[idx]
        label = self._label[idx]
        return img, label

    def __len__(self):
        return len(self._data)


class genTestSet(Dataset):
    def __init__(self):
        data = np.load('./dataset/data.npy')[140:]
        label = np.load('./dataset/label.npy')[140:]
        # Xt, X, yt, y = train_test_split(data, label, test_size=0.2, shuffle=True, random_state=70)
        # Xv, Xs, yv, ys = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=70)
        self._data = data
        self._label = label

    def __getitem__(self,idx):
        # Xtrain, Xtest, ytrain, ytest = train_test_split(self._data, self._label, test_size=0.1, shuffle=True, random_state=70)
        img = self._data[idx]
        label = self._label[idx]
        return img, label

    def __len__(self):
        return len(self._data)


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                        0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * (
                                    (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)


class LossAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)
