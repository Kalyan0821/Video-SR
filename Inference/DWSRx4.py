""" Author's inference script modified """

from __future__ import print_function
import os, glob, time
import tensorflow as tf
import numpy as np
from netx4 import model
import cv2
import pywt as pw
import ntpath
import argparse

TEST_EXPERIMENT = True

parser = argparse.ArgumentParser()
parser.add_argument('saved_model_path', help='Saved model path, of the form: xyz.ckpt')
parser.add_argument('path_in', help='upsized_Y_path (directory)')
parser.add_argument('path_out', help='sr_Y_path (directory)')
args = parser.parse_args()

TEST_MODEL_PATH = args.saved_model_path
TEST_SAVE_PATH = args.path_out
TEST_PATH = args.path_in

WV = 'db1'

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


if __name__ == '__main__':
    test_input = tf.placeholder(np.float32)
    # Feeding Forward
    test_output, _, _ = model(test_input)
    with tf.Session(config=tf.ConfigProto()) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        if TEST_EXPERIMENT:
            print ('>>>>>>>>Resuming Experiments For Testing')
            saver.restore(sess, TEST_MODEL_PATH)
            if not os.path.exists(str(TEST_SAVE_PATH)):
                os.makedirs(str(TEST_SAVE_PATH))
            for testImgName in glob.glob(TEST_PATH + '*.png'):
                print('Test Image %s' % path_leaf(testImgName))
                testBBImg = cv2.imread(testImgName, 0)
                tcoeffs = pw.dwt2(testBBImg, WV)
                tcA, (tcH, tcV, tcD) = tcoeffs
                tcA = tcA.astype(np.float32) / 255
                tcH = tcH.astype(np.float32) / 255
                tcV = tcV.astype(np.float32) / 255
                tcD = tcD.astype(np.float32) / 255
                test_temp = np.array([tcA, tcH, tcV, tcD])
                test_elem = np.rollaxis(test_temp, 0, 3)
                test_data = test_elem[np.newaxis, ...]
                start_time = time.time()
                output_data = sess.run([test_output], feed_dict={test_input: test_data})
                duration = time.time() - start_time
                dcA = output_data[0][0, :, :, 0]
                dcH = output_data[0][0, :, :, 1]
                dcV = output_data[0][0, :, :, 2]
                dcD = output_data[0][0, :, :, 3]
                srcoeffs = (dcA * 255 + tcA * 255,
                            (dcH * 255 + tcH * 255,
                             dcV * 255 + tcV * 255,
                             dcD * 255 + tcD * 255))
                sr_img = pw.idwt2(srcoeffs, WV)
                cv2.imwrite(str(TEST_SAVE_PATH +
                                str(path_leaf(testImgName))), sr_img)
                print ('Image %s, processing time %s' % (path_leaf(testImgName),
                                                       str(duration)))
        else:
            print('>>>>>>>>>Wrong script for New Experiment!')
    sess.close()
    print('Y channel inference complete')

