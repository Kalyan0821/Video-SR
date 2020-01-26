import cv2
import os
import numpy as np
import argparse
from PIL import Image

SR_SIZE = (1080, 1920)  # 1080p target
m_train = 768  # Div2k

parser = argparse.ArgumentParser()
parser.add_argument('path_in', help='uncropped_source_dir')
parser.add_argument('path_out', help='cropped_destination_dir')
parser.add_argument('window', help='crop_size', type=int, choices=[40, 80])
parser.add_argument('ov', help='overlap', type=int, choices=[0, 10, 20])
args = parser.parse_args()

uncropped_source_dir = args.path_in
cropped_destination_dir = args.path_out
w = args.window # 40
overlap = args.ov  # 10

def CropAndSave(uncropped_Y, filename):
	stride = w - overlap
	crop_index = 0
	for y in range(0, SR_SIZE[0]-w, stride):
		for x in range(0, SR_SIZE[1]-w, stride):
			crop = uncropped_Y[y:y+w, x:x+w]
			name, ext = os.path.splitext(filename)
			crop_index += 1
			cropped_filename = name + '_c{}'.format(crop_index) + ext
			cropped_Y_path = os.path.join(cropped_destination_dir, cropped_filename)
			cv2.imwrite(cropped_Y_path, crop)

	return crop_index
	

i = 1
for filename in sorted(os.listdir(uncropped_source_dir)): # Y channels
	uncropped_Y_path = os.path.join(uncropped_source_dir, filename)
	uncropped_Y = cv2.imread(uncropped_Y_path)
	crop_index = CropAndSave(uncropped_Y, filename)

	print("Image {}/{} done, {} crops generated".format(i, m_train, crop_index))
	i += 1

