
import os
import re
import cv2
import numpy as np
from collections import Counter
from optparse import OptionParser
from matplotlib import pyplot as plt
from skimage import img_as_float, img_as_uint
import copy
from PIL import Image

import sys
import h5py
import itertools

# python tools for our lf database
import file_io
# additional light field tools
import lf_tools


def create_hdf5_container(path_i1, path_o, lf_name):

	px = 48
	py = 48

	nviews = 9

	sx = 16
	sy = 16

	file = h5py.File(path_o + '/' + lf_name + '.hdf5', 'w')

	# read diffuse color
	LF = file_io.read_lightfield(path_i1)
	LF = LF.astype(np.float32)  # / 255.0

	cv_gt = lf_tools.cv(LF)
	lf_tools.save_image(path_o + '/' + lf_name, cv_gt)

	# maybe we need those, probably not.
	param_dict = file_io.read_parameters(path_i1)

	dset_blocks = []
	# block count: write out one individual light field
	cx = np.int32((LF.shape[3] - px) / sx) + 1
	cy = np.int32((LF.shape[2] - py) / sy) + 1

	for i, j in itertools.product(np.arange(0, nviews), np.arange(0, nviews)):
		dset_blocks.append(file.create_dataset('views%d%d' % (i, j), (cy, cx, 3, px, py),
											   chunks=(1, 1, 3, px, py),
											   maxshape=(None, None, 3, px, py)))
	# lists indexed in 2D
	dset_blocks = [dset_blocks[x:x + nviews] for x in range(0, len(dset_blocks), nviews)]

	sys.stdout.write(lf_name + ': ')

	for bx in np.arange(0, cx):
		sys.stdout.write('.')
		sys.stdout.flush()

		for by in np.arange(0, cy):

			x = bx * sx
			y = by * sx

			# extract data
			for i, j in itertools.product(np.arange(0, nviews), np.arange(0, nviews)):
				dset_blocks[i][j][bx, by, :, :, :] = np.transpose(np.array(LF[i, j, x:x + px, y:y + py, :]), (-1, 0, 1)).reshape(3, px, py)



	sys.stdout.write(' Done.\n')



def main():

	path_i = '/media/bwijerat/WinStorage/bwijerat/Documents/Lnx/EECS6400/LF-intrinsics-data/1'

	dir_o = 'OUTPUT_dataset'
	path_o = os.path.join(path_i, dir_o)

	if not os.path.exists(path_o):
		os.mkdir(path_o)
		print("Directory ", dir_o, " created")
	else:
		print("Directory ", dir_o, " already exists")


	dir_names = sorted([dI for dI in os.listdir(path_i) if os.path.isdir(os.path.join(path_i, dI))],
					   key=lambda s: s.casefold())

	dir_names.pop(dir_names.index('OUTPUT_hdf5'))

	for lf_name in dir_names:

		path_i1 = os.path.join(path_i, lf_name)
		create_hdf5_container(path_i1, path_o, lf_name)






if __name__== "__main__":
	main()