import tensorflow as tf
import numpy as np
import cv2
import pywt as pw
from netx4 import model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

###########################  HYPERPARAMETERS  ########################################

TO_RESTORE_FROM_EXISTING_CHECKPOINT = True

ALPHA0 = 1e-4  # initial learning rate
MINI_BATCH_SIZE = 64
m_train = 2386540  # total number of training pairs (cropped_hr, cropped_gt) in Div2k

TO_CLIP_GRADIENTS = False
GRADIENT_CLIP = 0.01  # clips only if above variable is set to True

LR_DECAY_RATE = 0.75

CROP_SIZE = (40, 40)
LAMBDA = 0.001
EPOCHS = 100

B = int(m_train/MINI_BATCH_SIZE)+1  # number of mini-batches/iterations per epoch
WV = 'db1'  # Wavelet function specification: Haar Wavelet

###########################  PATHS  ########################################

INITIAL_MODEL_PATH = 'Weightx4/x4.ckpt'  # parameters will be initialized with these PRE-TRAINED weights
TRAIN_CHECKPOINTS_PATH = 'train_checkpoints/x6.ckpt'  # checkpoints will be saved here during training
cropped_destination_dir = 'Train_cropped_set/'  # 40x40 crops of the Y channels must be present in this directory, within
# sub-directories named 'HR_Y_crops' and 'GT_Y_crops'
COSTS_SAVE_PATH = 'train_costs/'  # save costs as csv files to this directory, every 1000 epochs

###########################  Build Graph  ########################################

X_HR_T = tf.placeholder(tf.float32, name='cnn_in')
R_T = tf.placeholder(tf.float32, name='cnn_desired_out')

out, weight_tensors, _ = model(X_HR_T)  # actual output tensor
print(out.shape)

J_unreg = 0.5*tf.reduce_sum(tf.square(R_T - out))
# L2 regularization
s = 0
for weight_tensor in weight_tensors[::2]:
    s += tf.reduce_sum(tf.square(weight_tensor))
J_reg = LAMBDA*s

J = J_unreg + J_reg

alpha = tf.placeholder(tf.float32, name='learning_rate')
if TO_CLIP_GRADIENTS:
    optimizer = tf.train.AdamOptimizer(alpha)
    grads_and_vars = optimizer.compute_gradients(J)  # [(grad1, var1), (grad2, var2), ...]
    grads, variables = zip(*grads_and_vars)  # grads: (grad1, grad2, ...)
    clipped_grads, global_norm = tf.clip_by_global_norm(grads, GRADIENT_CLIP)  # clipped_grads : [grad1, grad2, ...]
    clipped_grads_and_vars = zip(clipped_grads, variables)
    minimizer = optimizer.apply_gradients(clipped_grads_and_vars)
else:
    minimizer = tf.train.AdamOptimizer(alpha).minimize(J)

datagen = ImageDataGenerator(data_format='channels_last')

hr_iterator = datagen.flow_from_directory(
	cropped_destination_dir,
	target_size=CROP_SIZE,
	color_mode='grayscale',
	classes=['HR_Y_crops'],
	class_mode=None,
	batch_size=MINI_BATCH_SIZE,
	shuffle=False,
	seed=1
	)  # next(hr_iterator) will yield a batch of size: 64x40x40x1
gt_iterator = datagen.flow_from_directory(
	cropped_destination_dir,
	target_size=CROP_SIZE,
	color_mode='grayscale',
	classes=['GT_Y_crops'],
	class_mode=None,
	batch_size=MINI_BATCH_SIZE,
	shuffle=False,
	seed=1
	)  # next(gt_iterator) will yield a batch of size: 64x40x40x1

###########################   Functions to compute DWT and IDWT on batches  ########################################

def DWT(images, WV): # images: (YxX) OR mx(YxX)
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)  # i.e. m=1
    if len(images.shape) == 3:
        images_A, (images_H, images_V, images_D) = pw.dwt2(images, WV)  # each: mx(Y'xX')
        transforms = np.stack((images_A, images_H, images_V, images_D), axis=3)  
        return transforms   # transforms: mx(Y'xX'x4)

def  IDWT(outputs, WV): # outputs: mx(Y'xX'x4)
    ouputs_A = outputs[:, :, :, 0]
    ouputs_H = outputs[:, :, :, 1]
    ouputs_V = outputs[:, :, :, 2]
    ouputs_D = outputs[:, :, :, 3]  # each: mx(Y'xX')
    coeffs = ouputs_A, (ouputs_H, ouputs_V, ouputs_D)
    inverses = pw.idwt2(coeffs, WV)  # inverses: mx(YxX)
    return inverses

###########################  Train  ########################################

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=30)

with tf.Session() as sess:
    sess.run(init)
    if TO_RESTORE_FROM_EXISTING_CHECKPOINT:
    	saver.restore(sess, INITIAL_MODEL_PATH)
    
    alpha_val = ALPHA0
    for e in range(1, EPOCHS+1):
        if e % 20 == 0:
            alpha_val *= LR_DECAY_RATE  # LR-decay
        costs_e = []

        for b in range(1, B+1):

            X_HR_val = next(hr_iterator)[:, :, :, 0]  # cropped HR minibatch, '_val' means evaluated value
            X_GT_val = next(gt_iterator)[:, :, :, 0]  # cropped GT minibatch
            assert X_HR_val.shape == X_GT_val.shape, '{}, {}'.format(X_HR_val.shape, X_GT_val.shape)
            
            X_HR_T_val = DWT(X_HR_val, WV)  # '_T_' means transform
            X_GT_T_val = DWT(X_GT_val, WV)

            R_T_val = X_GT_T_val - X_HR_T_val  # residual

            X_HR_T_val /= 255
            R_T_val /= 255
            
            # Assign placeholder values to feed into the graph
            placeholder_values = {X_HR_T: X_HR_T_val,
                                  R_T: R_T_val,
                                  alpha: alpha_val}
            _, J_val = sess.run([minimizer, J], feed_dict=placeholder_values)
            
            
            if b % 1000 == 0 or b == 1:
                print('Epoch {}: Cost after {} iterations = {}'.format(e, b, J_val))
                costs_e.append(J_val)
        np.savetxt(os.path.join(COSTS_SAVE_PATH, 'costs_{}.csv'.format(e)), costs_e, delimiter=',')

        saver.save(sess, TRAIN_CHECKPOINTS_PATH, global_step=e)
        print('Epoch {} done: Parameters have been saved\n=================='.format(e))

        