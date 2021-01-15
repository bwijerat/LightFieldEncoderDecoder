
from __future__ import print_function, division
import os
import sys
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import itertools
import pickle
import h5py


class LFAutoEncoder(nn.Module):

    def __init__(self):

        super(LFAutoEncoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # ENCODER #######################################################################
        #################################################################################

        # Assume same number of channels per input branch
        self.e_N_ch = 12

        self.e_in_ch = 1                  # number of Decoder input channels
        self.e_out_ch = self.e_N_ch         # number of output channels per conv layer, arbitrarily chosen
        self.e_N = 2                      # number of re-use per conv layer

        # ENCODER
        self.e_bn1 = nn.BatchNorm2d(self.e_N_ch)
        self.e_bn2 = nn.BatchNorm2d(self.e_N_ch)
        self.e_bn3 = nn.BatchNorm2d(self.e_N_ch)
        self.e_bn4 = nn.BatchNorm2d(self.e_N_ch)

        # [0][0]-------------------------------------------------------------------------------------------------------
        self.e_conv00 = nn.Conv2d(in_channels=self.e_out_ch, out_channels=self.e_out_ch, kernel_size=3,
                                stride=1, padding=1, groups=self.e_out_ch)

        self.e_bn1_00a = nn.BatchNorm2d(self.e_out_ch)
        self.e_bn1_00b = nn.BatchNorm2d(self.e_out_ch)
        # [0][1]-------------------------------------------------------------------------------------------------------
        self.e_conv01 = nn.Conv2d(in_channels=self.e_out_ch, out_channels=self.e_out_ch, kernel_size=3,
                                stride=1, padding=1, groups=self.e_out_ch)

        self.e_bn1_01a = nn.BatchNorm2d(self.e_out_ch)
        self.e_bn1_01b = nn.BatchNorm2d(self.e_out_ch)
        # [1][0]-------------------------------------------------------------------------------------------------------
        self.e_conv10 = nn.Conv2d(in_channels=self.e_out_ch, out_channels=self.e_out_ch, kernel_size=3,
                                stride=1, padding=1, groups=self.e_out_ch)

        self.e_bn1_10a = nn.BatchNorm2d(self.e_out_ch)
        self.e_bn1_10b = nn.BatchNorm2d(self.e_out_ch)
        # [1][1]-------------------------------------------------------------------------------------------------------
        self.e_conv11 = nn.Conv2d(in_channels=self.e_out_ch, out_channels=self.e_out_ch, kernel_size=3,
                                stride=1, padding=1, groups=self.e_out_ch)

        self.e_bn1_11a = nn.BatchNorm2d(self.e_out_ch)
        self.e_bn1_11b = nn.BatchNorm2d(self.e_out_ch)

        # Main  -------------------------------------------------------------------------------------------------------

        e_conv2_ch = 4*self.e_N*self.e_out_ch
        self.e_conv2 = nn.Conv2d(in_channels=e_conv2_ch, out_channels=e_conv2_ch, kernel_size=3,
                               stride=1, padding=1, groups=e_conv2_ch)
        self.e_bn2_a = nn.BatchNorm2d(e_conv2_ch)
        self.e_bn2_b = nn.BatchNorm2d(e_conv2_ch)

        # Encoded output to 3 channels.Cannot re-use layer, input ch != output ch.
        # Cannot re-use layer, input ch != output ch.
        e_conv3_ch = 4*self.e_N*self.e_out_ch + self.e_N*e_conv2_ch
        self.e_conv3 = nn.Conv2d(in_channels=e_conv3_ch, out_channels=self.e_in_ch, kernel_size=3,
                               stride=1, padding=1, groups=self.e_in_ch)
        self.e_bn3_a = nn.BatchNorm2d(self.e_in_ch)
        self.e_bn3_b = nn.BatchNorm2d(self.e_in_ch)

        # DECODER #######################################################################
        #######################################################################

        self.d_in_ch = self.e_in_ch   # number of Decoder input channels
        self.d_N = self.e_N           # number of re-use per conv layer
        self.d_N_ch = self.e_N_ch      # number of decoder output channels per branch

        # DECODER
        self.d_conv1 = nn.Conv2d(in_channels=self.d_in_ch, out_channels=self.d_in_ch, kernel_size=3,
                               stride=1, padding=1, groups=self.d_in_ch)
        self.d_bn1_a = nn.BatchNorm2d(self.d_in_ch)
        self.d_bn1_b = nn.BatchNorm2d(self.d_in_ch)
        self.d_bn1_c = nn.BatchNorm2d(self.d_in_ch)
        self.d_bn1_d = nn.BatchNorm2d(self.d_in_ch)

        d_conv_2_ch = 4 * self.d_in_ch
        d_conv_3_ch = 4 * self.d_in_ch + self.d_N * d_conv_2_ch

        # [0][0]-------------------------------------------------------------------------------------------------------
        self.d_conv00_2 = nn.Conv2d(in_channels=d_conv_2_ch, out_channels=d_conv_2_ch,
                                  kernel_size=3, stride=1, padding=1, groups=d_conv_2_ch)
        self.d_bn2_00a = nn.BatchNorm2d(d_conv_2_ch)
        self.d_bn2_00b = nn.BatchNorm2d(d_conv_2_ch)

        # Cannot re-use layer, input ch != output ch.
        self.d_conv00_3 = nn.Conv2d(in_channels=d_conv_3_ch, out_channels=self.d_N_ch,
                                  kernel_size=3, stride=1, padding=1, groups=self.d_N_ch)
        self.d_bn3_00a = nn.BatchNorm2d(self.d_N_ch)

        # [0][1]-------------------------------------------------------------------------------------------------------
        self.d_conv01_2 = nn.Conv2d(in_channels=d_conv_2_ch, out_channels=d_conv_2_ch,
                                  kernel_size=3, stride=1, padding=1, groups=d_conv_2_ch)
        self.d_bn2_01a = nn.BatchNorm2d(d_conv_2_ch)
        self.d_bn2_01b = nn.BatchNorm2d(d_conv_2_ch)

        # Cannot re-use layer, input ch != output ch.
        self.d_conv01_3 = nn.Conv2d(in_channels=d_conv_3_ch, out_channels=self.d_N_ch,
                                  kernel_size=3, stride=1, padding=1, groups=self.d_N_ch)
        self.d_bn3_01a = nn.BatchNorm2d(self.d_N_ch)

        # [1][0]-------------------------------------------------------------------------------------------------------
        self.d_conv10_2 = nn.Conv2d(in_channels=d_conv_2_ch, out_channels=d_conv_2_ch,
                                  kernel_size=3, stride=1, padding=1, groups=d_conv_2_ch)
        self.d_bn2_10a = nn.BatchNorm2d(d_conv_2_ch)
        self.d_bn2_10b = nn.BatchNorm2d(d_conv_2_ch)

        # Cannot re-use layer, input ch != output ch.
        self.d_conv10_3 = nn.Conv2d(in_channels=d_conv_3_ch, out_channels=self.d_N_ch,
                                  kernel_size=3, stride=1, padding=1, groups=self.d_N_ch)
        self.d_bn3_10a = nn.BatchNorm2d(self.d_N_ch)

        # [1][1]-------------------------------------------------------------------------------------------------------
        self.d_conv11_2 = nn.Conv2d(in_channels=d_conv_2_ch, out_channels=d_conv_2_ch,
                                  kernel_size=3, stride=1, padding=1, groups=d_conv_2_ch)
        self.d_bn2_11a = nn.BatchNorm2d(d_conv_2_ch)
        self.d_bn2_11b = nn.BatchNorm2d(d_conv_2_ch)

        # Cannot re-use layer, input ch != output ch.
        self.d_conv11_3 = nn.Conv2d(in_channels=d_conv_3_ch, out_channels=self.d_N_ch,
                                  kernel_size=3, stride=1, padding=1, groups=self.d_N_ch)
        self.d_bn3_11a = nn.BatchNorm2d(self.d_N_ch)


    def forward(self, x):

        enc = self.encode(x[0, :, : , :], x[1, :, : , :], x[2, :, : , :], x[3, :, : , :])
        dec = self.decode(enc)

        return enc, dec

    def encode(self, x):

        e_bn1 = self.relu(self.e_bn1(x))
        e_bn2 = self.relu(self.e_bn2(x))
        e_bn3 = self.relu(self.e_bn3(x))
        e_bn4 = self.relu(self.e_bn4(x))

        # [0]01]
        e_conv00a = self.relu(self.e_bn1_00a(self.e_conv00(e_bn1)))
        e_conv00b = self.relu(self.e_bn1_00b(self.e_conv00(e_conv00a)))
        # [0][1]
        e_conv01a = self.relu(self.e_bn1_01a(self.e_conv01(e_bn2)))
        e_conv01b = self.relu(self.e_bn1_01b(self.e_conv01(e_conv01a)))
        # [1][0]
        e_conv10a = self.relu(self.e_bn1_10a(self.e_conv10(e_bn3)))
        e_conv10b = self.relu(self.e_bn1_10b(self.e_conv10(e_conv10a)))
        # [1][1]
        e_conv11a = self.relu(self.e_bn1_11a(self.e_conv11(e_bn4)))
        e_conv11b = self.relu(self.e_bn1_11b(self.e_conv11(e_conv11a)))


        # Concatenate in channel dimension
        e_c2_dense = self.relu(torch.cat([  e_conv00a, e_conv00b,
                                            e_conv01a, e_conv01b,
                                            e_conv10a, e_conv10b,
                                            e_conv11a, e_conv11b], 1))

        e_conv2a = self.relu(self.e_bn2_a(self.e_conv2(e_c2_dense)))
        e_conv2b = self.relu(self.e_bn2_b(self.e_conv2(e_conv2a)))

        e_c3_dense = self.relu(torch.cat([  e_c2_dense,
                                            e_conv2a, e_conv2b], 1))

        # No ReLU after final batch normalization to increase encoded efficiency
        enc = self.e_bn3_a(self.e_conv3(e_c3_dense))


        return enc

    def decode(self, x):
        enc_x = self.relu(x)
        d_conv1a = self.relu(self.d_bn1_a(self.d_conv1(enc_x)))
        d_conv1b = self.relu(self.d_bn1_b(self.d_conv1(d_conv1a)))
        d_conv1c = self.relu(self.d_bn1_c(self.d_conv1(d_conv1b)))
        d_conv1d = self.relu(self.d_bn1_d(self.d_conv1(d_conv1c)))

        # Concatenate in channel dimension
        d_c2_dense = self.relu(torch.cat([enc_x, d_conv1a, d_conv1b, d_conv1c, d_conv1d], 1))

        # [0][0]-------------------------------------------------------------------------------------------------------
        d_conv00_2a = self.relu(self.d_bn2_00a(self.d_conv00_2(d_c2_dense)))
        d_conv00_2b = self.relu(self.d_bn2_00b(self.d_conv00_2(d_conv00_2a)))

        # Concatenate in channel dimension
        d_c3_dense = self.relu(torch.cat([d_c2_dense, d_conv00_2a, d_conv00_2b], 1))

        d_conv00_3a = self.relu(self.d_bn3_00a(self.d_conv00_3(d_c3_dense)))

        output00 = nn.Sigmoid(d_conv00_3a)

        # [0][1]-------------------------------------------------------------------------------------------------------
        d_conv01_2a = self.relu(self.d_bn2_01a(self.d_conv01_2(d_c2_dense)))
        d_conv01_2b = self.relu(self.d_bn2_01b(self.d_conv01_2(d_conv01_2a)))

        # Concatenate in channel dimension
        d_c3_dense = self.relu(torch.cat([d_c2_dense, d_conv01_2a, d_conv01_2b], 1))

        d_conv01_3a = self.relu(self.d_bn3_01a(self.d_conv01_3(d_c3_dense)))

        output01 = nn.Sigmoid(d_conv01_3a)

        # [1][0]-------------------------------------------------------------------------------------------------------
        d_conv10_2a = self.relu(self.d_bn2_10a(self.d_conv10_2(d_c2_dense)))
        d_conv10_2b = self.relu(self.d_bn2_10b(self.d_conv10_2(d_conv10_2a)))

        # Concatenate in channel dimension
        d_c3_dense = self.relu(torch.cat([d_c2_dense, d_conv10_2a, d_conv10_2b], 1))

        d_conv10_3a = self.relu(self.d_bn3_10a(self.d_conv10_3(d_c3_dense)))

        output10 = nn.Sigmoid(d_conv10_3a)

        # [1][1]-------------------------------------------------------------------------------------------------------
        d_conv11_2a = self.relu(self.d_bn2_11a(self.d_conv11_2(d_c2_dense)))
        d_conv11_2b = self.relu(self.d_bn2_11b(self.d_conv11_2(d_conv11_2a)))

        # Concatenate in channel dimension
        d_c3_dense = self.relu(torch.cat([d_c2_dense, d_conv11_2a, d_conv11_2b], 1))

        d_conv11_3a = self.relu(self.d_bn3_11a(self.d_conv11_3(d_c3_dense)))

        output11 = nn.Sigmoid(d_conv11_3a)

        # DECODED
        dec = torch.cat([output00, output01, output10, output11], 1)

        return dec



class LFAutoEncoderDataset(Dataset):

    def __init__(self, data_root, imgdim, nviews, npatches, subviewsize, samplesize):

        self.data_path = data_root

        self.img_dim = imgdim
        self.nviews = nviews
        self.npatches = npatches


        # size of the subview (sample_size & sample_range sample from it, replacing it during iteration)
        self.subview_size = subviewsize
        # range subview should move
        self.subview_range = np.array([nviews, nviews]) - (subviewsize - np.array([1, 1]))

        # size of sampling view
        self.sample_size = samplesize
        # range sampling view should move
        self.sample_range = self.subview_size - (samplesize - np.array([1, 1]))

        # names of LF in dataset
        self.db_LF_names = sorted([dI for dI in os.listdir(self.data_path) if dI.endswith('hdf5')],
                                  key=lambda s: s.casefold())
        # number of samples in dataset
        self.samples_len = len(self.db_LF_names) * self.npatches * self.npatches * self.subview_range[0] * \
                           self.subview_range[1]
        # number of subviews
        self.subviews_len = self.sample_range[0]*self.sample_range[1]
        # number of views
        self.views_len = self.sample_size[0]*self.sample_size[1]

        # LF Sampling Database
        self.db_sp_path = os.path.dirname(data_root)
        self.db_sp_filename = '_LF_sampling_scheme'
        self.db_sp_label = 'sampling-scheme'
        self.db_sp = []

        # LF Database
        self.db_LF_names = []
        self.db_LF_label = 'patch-view-sample'
        self.db_LF = []

        # __init__(): names of LFs and access hdf5 files containing LF data
        self.db_LF_names = sorted([dI for dI in os.listdir(self.data_path) if dI.endswith('hdf5')],
                                  key=lambda s: s.casefold())
        sys.stdout.write('Fetching LF data.. '); sys.stdout.flush()
        self.db_LF = []
        for LF_name in self.db_LF_names:
            sys.stdout.write(LF_name + ' '); sys.stdout.flush()
            self.db_LF.append(h5py.File(os.path.join(self.data_path, LF_name), 'r')[self.db_LF_label])
        sys.stdout.write('Done. \n');

    def __len__(self):

        return self.samples_len

    def __getitem__(self, idx):

        return self.get_sample(idx)


    def init_dataset(self):

        sys.stdout.write('Initializing Sampling Scheme: \n')
        sys.stdout.flush()


        # HDF5 file and database creation
        file = h5py.File(os.path.join(self.db_sp_path, self.db_sp_filename) + '.hdf5', 'w')

        self.db_sp = file.create_dataset(self.db_sp_label,
                                    (self.samples_len, self.sample_range[0], self.sample_range[1], self.sample_size[0], self.sample_size[1], 5),
                                    chunks=(1, self.sample_range[0], self.sample_range[1], self.sample_size[0], self.sample_size[1], 5),
                                    maxshape=(None, None, None, None, None, 5),
                                    dtype=np.uint8)

        cnt = 0
        for LF_idx, LF_name in  enumerate(self.db_LF_names, start=0):

            sys.stdout.write("({}/{}) {}: ".format(LF_idx + 1, len(self.db_LF_names), LF_name))
            sys.stdout.flush()

            # Range of patches
            for bx in np.arange(0, self.npatches):

                sys.stdout.write('.')
                sys.stdout.flush()

                for by in np.arange(0, self.npatches):

                    # Range of subview
                    for i1, j1 in itertools.product(np.arange(0, self.subview_range[0]), np.arange(0, self.subview_range[1])):

                        # Range of sample
                        for m1, n1 in itertools.product(np.arange(0, self.sample_range[0]), np.arange(0, self.sample_range[1])):

                            # Sampling
                            for m2, n2 in itertools.product(np.arange(0, self.sample_size[0]), np.arange(0, self.sample_size[1])):

                                self.db_sp[cnt, m1, n1, m2, n2, :] = LF_idx, bx, by, i1+m1+m2, j1+n1+n2

                        cnt = cnt + 1

            sys.stdout.write(' Done.\n')
            sys.stdout.flush()


    def get_sampling(self):

        sys.stdout.write('Loading LF sampling scheme.. '); sys.stdout.flush()
        self.db_sp = h5py.File(os.path.join(self.db_sp_path, self.db_sp_filename) + '.hdf5', 'r')[self.db_sp_label]
        sys.stdout.write('Done. \n')



    def get_sample(self, id):

        sp = self.db_sp[id]                    # establish sampling scheme
        db_LF = self.db_LF[sp[0, 0, 0, 0, 0]]   # establish LF to sample from

        C, H, W = self.img_dim[0], self.img_dim[1], self.img_dim[2]

        # Tensor to hold all subviews
        subviews_tensor = torch.zeros([self.subviews_len, self.views_len * C, H, W], dtype=torch.float32)

        idx = 0
        for m1, n1 in itertools.product(np.arange(0, self.sample_range[0]), np.arange(0, self.sample_range[1])):
            idy = 0
            for m2, n2 in itertools.product(np.arange(0, self.sample_size[0]), np.arange(0, self.sample_size[1])):

                LF_idx, bx, by, vx, vy = sp[m1, n1, m2, n2, :]
                subviews_tensor[idx, (idy * C):((idy * C) + C), :, :] = torch.Tensor(db_LF[bx, by, vx, vy, :, :, :])
                idy = idy + 1

            idx = idx + 1

        return subviews_tensor



