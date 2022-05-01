import os
import sys
import PIL.Image
import ut
import torch
import weights_init
import numpy as np
import torch.nn as nn

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
    train_loader = DataLoader(dataset=ut.genDataSet(), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=ut.genValidationSet(), batch_size=batch_size, shuffle=True)


    net = UNet(n_channels=8, n_classes=2)
    net.apply(weights_init.init_model)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.01)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.ones(2, 384, 384))

    model_save_path = './model/test.pth'
    if_loadmodel = True
    if_savemodel = True
    if Path(model_save_path).exists() and if_loadmodel:
        net.load_state_dict(torch.load(model_save_path))

    trainlossSet, testlossSet, scoreSet, all_pre, epochSet, all_lab = ([] for _ in range(6))
    best_loss = float('inf')
    for epoch in range(1, 50):
        train_loss, test_loss, i, j = 0.0, 0.0, 0, 0
        for _, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            i += 1
            # x = x.cuda().float()
            # y = y.cuda().long()
            x = x.squeeze(1).float()
            y = y.squeeze(1).long()
            y_ = net(x)
            cost = loss(y_, y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            train_loss += cost.item()
        trainlossSet.append(train_loss / i)
        # print("Training %d epoch... " % (epoch))

        if epoch % 1 == 0:
            with torch.no_grad():
                net.eval()
                for _, (x, y) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
                    j += 1
                    # X = X.cuda().float()
                    # Y = Y.cuda()
                    x = x.squeeze(1).float()
                    y = y.squeeze(1).long()
                    Y_ = net(x)
                    COST = loss(Y_, y)
                    test_loss += COST.item()
                    Y_ = nn.functional.softmax(Y_, 1).argmax(1)

                    all_lab.append(y.view(batch_size, -1).detach().cpu().numpy())
                    all_pre.append(Y_.view(batch_size, -1).detach().cpu().numpy())
                    # 将每个Batch的预测值和标签加入list组成矩阵，便于指标计算
                cur_loss = test_loss / j
                testlossSet.append(cur_loss)
                all_Y = np.concatenate(all_lab)
                all_Y_ = np.concatenate(all_pre)

            # 计算三类指标
            every_class, confusion_mat = ut.CalAccuracy(all_Y, all_Y_, 2)
            acc = every_class[:]
            print("%d epoch validation loss: %.4f" %(epoch, cur_loss))
            print("Acc background: %.2f\t object:%.2f\tOA: %.2f\tAA: %.2f\tkappa: %.2f\t" %
                  (acc[0], acc[1], acc[2], acc[3], acc[4]))
            epochSet.append(epoch)
            scoreSet.append([acc[2], acc[3]])

            if if_savemodel and cur_loss < best_loss:
                torch.save(net.state_dict(), model_save_path)
                best_loss = cur_loss
                print("save...")

    print("Training end.")
