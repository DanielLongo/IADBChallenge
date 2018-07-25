import tensorflow as tf
from ReadData_debug import read_data
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from random import shuffle

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import time

def create_placeholders():
	placeholder_x = tf.placeholder(tf.float32, shape=(None, 140))
	placeholder_y = tf.placeholder(tf.float32, shape=(None, 4))
	return placeholder_x, placeholder_y


def foward_pass(x):
	# a1 = tf.contrib.layers.fully_connected(x, 100)
	a1 = tf.layers.dense(x, 100, activation=tf.nn.relu, name="h1", kernel_initializer=tf.zeros_initializer())
	# self.log_layer("h1")
	# a2 = tf.contrib.layers.fully_connected(a1, 64)
	a2 = tf.layers.dense(a1, 64, activation=tf.nn.relu, name="h2", kernel_initializer=tf.zeros_initializer())
	# self.log_layer("h2")
	# a3 = tf.contrib.layers.fully_connected(a2, 16)
	a3 = tf.layers.dense(a2, 16, activation=tf.nn.relu, name="h3", kernel_initializer=tf.zeros_initializer())
	# self.log_layer("h3")
	# a4 = tf.contrib.layers.fully_connected(a3, 4, activation_fn=tf.nn.sigmoid)
	a4 = tf.layers.dense(a3, 4, activation=tf.nn.sigmoid, name="h4", kernel_initializer=tf.zeros_initializer())

	return a4

def loss_op(preds, y):
	costs = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
	avg_cost = tf.reduce_mean(costs)
	return avg_cost

def train_NN(train_x, train_y, epoch, lr=.01, show_graph=True):
	tf.reset_default_graph()
	ops.reset_default_graph()
	placeholder_x, placeholder_y = create_placeholders()
	preds = foward_pass(placeholder_x)
	avg_cost = loss_op(preds, placeholder_y)
	optimizer = tf.train.AdamOptimizer(lr).minimize(avg_cost)
	initialize = tf.global_variables_initializer()

	costs = []
	batch_costs = []
	with tf.Session() as sess:
		sess.run(initialize)
		for i in range(1, epoch + 1):
			for b in range(len(train_x)):
				_, cur_cost, cur_preds = sess.run([optimizer, avg_cost, preds], {placeholder_x : train_x[b], placeholder_y : train_y[b]})
				# print("Current cost", cur_cost)
				if (i % 10 == 0) and (b == 1):
					print("Cost @", i, cur_cost)
				batch_costs += [cur_cost]
			costs += [sum(batch_costs)/len(batch_costs)]

		if show_graph:
			x = [i for i in range(len(costs))]
			y = costs

			plt.plot(x, y)
			plt.ylabel("Cost")
			plt.xlabel("Epoch")
			plt.show()

		return sess

def test_NN(test_x, test_y, sess):
	
	with tf.Session as sess:
		preds = tf.argmax

train_x, train_y, eval_x, eval_y, test_x, test_y = read_data("./data/train.csv")

# print("train_x", pd.isnull(np.asarray(train_x)).any())
# print("train_y", pd.isnull(np.asarray(train_y)).any())
# print("eval_x", pd.isnull(np.asarray(eval_x)).any())
# print("eval_y", pd.isnull(np.asarray(eval_y)).any())
# print("test_x", pd.isnull(np.asarray(test_x)).any())
# print("test_y", pd.isnull(np.asarray(test_y)).any())
start = time.time()
sess = train_NN(train_x, train_y, 100, lr=.001, show_graph=False)
print("FINISHED WITH Manual")
end = time.time()
print("total time", start - end)







# print("Finished 1")
# nn.train_NN(300, .01, show_graph=False)
# print("Finished 2")
# nn.train_NN(300, .00025, show_graph=False)
# print("Finished 3")
# nn.compare_runs(show_graph=False)
# print("FInished with All")
