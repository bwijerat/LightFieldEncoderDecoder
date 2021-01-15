
import torch
import torchvision as tv
from torchvision import transforms
import torch.nn as nn
import torch.nn.init as init
import numpy as np

import time
import copy
import cv2
import os
import sys
import h5py
import itertools
import pandas as pd

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import DenseNetwork as dn
import other_tools as tools


# Hyper-Parameters
num_epochs = 5
batch_size = 16
learning_rate = 1e-3


# ########################################################
# CSV Logging via Pandas Dataframe
# ########################################################
logs = [dI for dI in os.listdir('results/') if dI.endswith(('txt','csv'))].__len__()
logs_archive = [dI for dI in os.listdir('results/archive/') if dI.endswith(('txt','csv'))].__len__()

csv_path = os.path.join('results/test_run%d.csv' % (logs + logs_archive + 1))
df_index = ['Sample Batch', 'Sampling Info']
for epoch in np.arange(1, num_epochs + 1):
    df_index.append('Epoch %d Loss' % epoch)

df = pd.DataFrame(columns=df_index)                     # pandas dataframe
df_loss = list(np.zeros(num_epochs, dtype = float))     # dataframe place holder for loss entries
df_time = list(np.zeros(num_epochs, dtype = int))


# ########################################################
# SAMPLING SCHEME
# ########################################################
nsamples = []
root_dir = '/home/bwijerat/Documents/EECS6400/Project/DATA/_OUTPUT_hdf5'
# root directory, img_dim (C, H, W), num views, sample size
dataset = dn.LFAutoEncoderDataset(root_dir, np.array([3, 48, 48]), 9, 30, np.array([3,3]), np.array([2,2]), nsamples)
# dataset.init_dataset()
dataset.get_sampling()


# ########################################################
# DEEP LEARNING SETUP
# ########################################################

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Send the model to GPU
model = dn.LFAutoEncoder3().cuda()
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = dn.LFAutoEncoder().to('cuda:0')

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.SmoothL1Loss()

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

model.apply(weight_init)


# ########################################################
# DATA VISUALIZATION
# ########################################################

results_path = '/home/bwijerat/Documents/EECS6400/Project/results'

def vis_compare(x, y, epoch):

    dir_path = os.path.join(results_path, ('test_run%d' % (logs + logs_archive + 1)))

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("Directory ", dir_path, " Created ")
    else:
        print("Directory ", dir_path, " already exists")

    dir_path = os.path.join(dir_path, ('epoch%d' % (epoch)))

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("Directory ", dir_path, " Created ")
    else:
        print("Directory ", dir_path, " already exists")

    inv_norm = dn.NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for id_batch, batch in enumerate(np.arange(0,x.shape[0]), start=0):
        x_= [x[batch, 0, :, :, :], x[batch, 1, :, :, :], x[batch, 2, :, :, :], x[batch, 3, :, :, :]]
        y_ = [y[batch, 0, :, :, :], y[batch, 1, :, :, :], y[batch, 2, :, :, :], y[batch, 3, :, :, :]]

        fig = plt.figure(id_batch)
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for id in np.arange(0,8,2):
            idx = int(id / 2)

            img_1x = torch.unsqueeze(inv_norm(x_[idx][0:3, :, :]), 0)
            img_2x = torch.unsqueeze(inv_norm(x_[idx][3:6, :, :]), 0)
            img_3x = torch.unsqueeze(inv_norm(x_[idx][6:9, :, :]), 0)
            img_4x = torch.unsqueeze(inv_norm(x_[idx][9:12, :, :]), 0)

            img_tensorx = torch.cat([img_1x, img_2x, img_3x, img_4x], 0)
            grid_imgx = (tv.utils.make_grid(img_tensorx, nrow=2))

            ax = fig.add_subplot(2, 4, (id+1))
            ax.imshow(grid_imgx.permute(1, 2, 0))

            img_1y = torch.unsqueeze(inv_norm(y_[idx][0:3, :, :]), 0)
            img_2y = torch.unsqueeze(inv_norm(y_[idx][3:6, :, :]), 0)
            img_3y = torch.unsqueeze(inv_norm(y_[idx][6:9, :, :]), 0)
            img_4y = torch.unsqueeze(inv_norm(y_[idx][9:12, :, :]), 0)

            img_tensory = torch.cat([img_1y, img_2y, img_3y, img_4y], 0)
            grid_imgy = (tv.utils.make_grid(img_tensory, nrow=2))

            ax = fig.add_subplot(2, 4, (id+2))
            ax.imshow(grid_imgy.permute(1, 2, 0))


        #fig.show()

        img_path = os.path.join(dir_path, ('sample%d.png' % (id_batch)))
        plt.savefig(img_path)


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    # random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

# deterministic randomization, same random order per epoch?
#random_seed(0, True)

df_out_list = []
len_dl = len(dataloader) - 1
for id_epoch, epoch in enumerate(range(num_epochs), start=0):
    start = time.time()
    for id_data, data in enumerate(dataloader, start=0):

        model.train()

        data = data.cuda()

        # ===================clear=====================
        optimizer.zero_grad()

        # ===================forward=====================

        enc, dec = model(data)

        loss = criterion(dec, data)

        # ===================backward====================
        loss.backward()
        optimizer.step()

        # ===================log========================
        out_log = [ ', '.join(map(str, nsamples)),
                    '[{}/{}] {:.2f}%'.format(id_data, len_dl, (id_data/len_dl)*100)]

        if (id_epoch == 0):
            df_loss[id_epoch] = round(loss.data.item(), 4)
            df.loc[id_data] = out_log + df_loss
        else:
            df_loss = df.loc[id_data][2:]
            df_loss[id_epoch] = round(loss.data.item(), 4)
            df.loc[id_data] = out_log + df_loss.tolist()

        df_out_list = df.loc[id_data].tolist()
        df_out_list.pop(0)

        sys.stdout.write(   '\nEpoch ({}/{}): '.format(epoch + 1, num_epochs) +
                            ' | '.join(map(str, df_out_list)))
        sys.stdout.flush()
        nsamples.clear()
        break

    end = time.time()
    df_time[id_epoch] = int(end-start)

    # Output pd dataframe to CSV file after every epoch
    df.to_csv(csv_path)
    # Output visual comparison after every epoch
    vis_compare(data.cpu().detach(), dec.cpu().detach(), id_epoch)


df_out_list[0] = 'Time(sec) per Epoch'
df_out_list[1:] =  df_time
df_out_list.insert(0, '')
df.loc[len_dl + 1] = df_out_list
df.to_csv(csv_path)





