import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path_in', help='lr_rgb_path')
parser.add_argument('path_out', help='sr_Y_path')
parser.add_argument('S', help='sr_factor', type=int, choices=[1, 4, 6])
parser.add_argument('final_destination', help='sr_rgb_path')
args = parser.parse_args()

lr_rgb_path = args.path_in
sr_Y_path = args.path_out
SCALE = args.S
sr_path = args.final_destination

def upsize(lr_rgb_image, SCALE, mode=cv2.INTER_CUBIC):
	n, m, c = lr_rgb_image.shape
	new_shape = (int(m*SCALE), int(n*SCALE))
	upsized_rgb_image = cv2.resize(lr_rgb_image, new_shape, interpolation=mode)
	return upsized_rgb_image

i = 1
for filename in os.listdir(sr_Y_path):
	lr_rgb_image_path = os.path.join(lr_rgb_path, filename)
	lr_rgb_image = cv2.imread(lr_rgb_image_path)

	if SCALE > 1:
		upsized_rgb_image = upsize(lr_rgb_image, SCALE)
	elif SCALE == 1:
		upsized_rgb_image = lr_rgb_image

	upsized_YCbCr_image = cv2.cvtColor(upsized_rgb_image, cv2.COLOR_BGR2YCR_CB)

	sr_Y_channel_path = os.path.join(sr_Y_path, filename)
	sr_Y_channel = cv2.imread(sr_Y_channel_path)
	upsized_YCbCr_image[:, :, 0] = sr_Y_channel[:, :, 0]

	sr_image_path = os.path.join(sr_path, filename)
	sr_image = cv2.cvtColor(upsized_YCbCr_image, cv2.COLOR_YCR_CB2BGR)
	cv2.imwrite(sr_image_path, sr_image)
	print("{} images converted".format(i))
	i += 1









