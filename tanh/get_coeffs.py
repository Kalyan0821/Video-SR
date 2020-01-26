import tensorflow as tf
import numpy as np

#def f(X, a, b, c, d):
#    term1 = tf.divide(tf.multiply(a, X), tf.add(1.0, tf.multiply(b, tf.abs(X)))) 
#    term2 = tf.divide(tf.multiply(c, X), tf.add(1.0, tf.multiply(d, tf.abs(X))))
#    return term1 - term2

def f(X, a):
    term1 = tf.divide(tf.multiply(a, X), tf.add(1.0, tf.multiply(a, tf.abs(X)))) 
    return term1

def tanh(X):
    return tf.math.tanh(X)

a = tf.Variable(10, dtype=tf.float32)  # init: 2.1
#b = tf.Variable(1.1, dtype=tf.float32)  # init: 1.1
#c = tf.Variable(0.7, dtype=tf.float32)  # init: 0.7
#d = b*c/(a - b)

X = tf.placeholder(tf.float32)

#squared_diff = tf.square(f(X, a, b, c, d) - tanh(X))
squared_diff = tf.square(f(X, a) - tanh(X))
J = tf.reduce_sum(squared_diff)  # cost

optimizer = tf.train.AdamOptimizer(1).minimize(J)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	x_values = np.array([0, 1, 0.001])  # Match functions from 0 to 1
	phv = {X: x_values}
	
	for i in range(10000):
		a_val, J_val = sess.run([a, J], feed_dict=phv)
		print(a_val)
		print("Cost:", J_val)

		sess.run(optimizer, feed_dict=phv)
		
print("\na = {}".format(a_val))

# Results:
# a = 3.19






