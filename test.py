import os
import sys
import PIL.Image
import ut
import numpy as np
import torch.nn as nn
import torch
import weights_init

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.unet_model import UNet
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    learning_rate = 0.0003
    batch_size = 2
    test_loader = DataLoader(dataset=ut.genTestSet(), batch_size=batch_size, shuffle=True)

    net = UNet(n_channels=8, n_classes=2)
    loss = nn.CrossEntropyLoss()

    model_save_path = './model/test.pth'
    if_loadmodel = True
    if Path(model_save_path).exists() and if_loadmodel:
        net.load_state_dict(torch.load(model_save_path))

    test_loss, i = 0.0, 0
    all_lab, all_pre = [], []
    net.eval()
    for idx, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
        i += 1
        x = x.squeeze(1).float()
        y = y.squeeze(1).long()
        y_ = net(x)
        cost = loss(y_, y)
        test_loss += cost.item()
        y_ = nn.functional.softmax(y_, 1).argmax(1)
        all_lab.append(y.view(batch_size, -1).detach().cpu().numpy())
        all_pre.append(y_.view(batch_size, -1).detach().cpu().numpy())
    all_Y = np.concatenate(all_lab)
    all_Y_ = np.concatenate(all_pre)
    every_class, confusion_mat = ut.CalAccuracy(all_Y, all_Y_, 2)
    acc = every_class[:]

    print("test loss %.4f" % (test_loss / i))
    print("acc background: %.2f\t object:%.2f\tOA: %.2f\tAA: %.2f\tkappa: %.2f\t" % (acc[0], acc[1], acc[2], acc[3], acc[4]))
    print(acc)

    ut.show_img(all_Y, all_Y_, 384)