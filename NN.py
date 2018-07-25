import tensorflow as tf
from ReadData_debug import read_data
import numpy as np
import pandas as pd
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import time

def create_placeholders():
	placeholder_x = tf.placeholder(tf.float32, shape=(None, 140))
	placeholder_y = tf.placeholder(tf.float32, shape=(None, 4))
	return placeholder_x, placeholder_y

def create_variables():
	w1 = tf.get_variable("W1", shape=[140, 100], initializer=tf.zeros_initializer())
	b1 = tf.get_variable("B1", shape=[100], initializer=tf.zeros_initializer())
	w2 = tf.get_variable("W2", shape=[100, 64], initializer=tf.zeros_initializer())
	b2 = tf.get_variable("B2", shape=[64], initializer=tf.zeros_initializer())
	w3 = tf.get_variable("W3", shape=[64, 16], initializer=tf.zeros_initializer())
	b3 = tf.get_variable("B3", shape=[16], initializer=tf.zeros_initializer())
	w4 = tf.get_variable("W4", shape=[16, 4], initializer=tf.zeros_initializer())
	b4 = tf.get_variable("B4", shape=[4], initializer=tf.zeros_initializer())
	# w1 = tf.get_variable("W1", shape=[140, 100], initializer=tf.contrib.layers.xavier_initializer())
	# b1 = tf.get_variable("B1", shape=[100], initializer=tf.contrib.layers.xavier_initializer())
	# w2 = tf.get_variable("W2", shape=[100, 64], initializer=tf.contrib.layers.xavier_initializer())
	# b2 = tf.get_variable("B2", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
	# w3 = tf.get_variable("W3", shape=[64, 16], initializer=tf.contrib.layers.xavier_initializer())
	# b3 = tf.get_variable("B3", shape=[16], initializer=tf.contrib.layers.xavier_initializer())
	# w4 = tf.get_variable("W4", shape=[16, 4], initializer=tf.contrib.layers.xavier_initializer())
	# b4 = tf.get_variable("B4", shape=[4], initializer=tf.contrib.layers.xavier_initializer())	

	params = [[w1,b1], [w2,b2], [w3,b3], [w4,b4]]
	return params

def foward_pass(x, params):
	w1, b1 = params[0][0], params[0][1]
	w2, b2 = params[1][0], params[1][1]
	w3, b3 = params[2][0], params[2][1]
	w4, b4 = params[3][0], params[3][1]

	z1 = tf.matmul(x, w1) + b1
	a1 = tf.nn.relu(z1)
	
	z2 = tf.matmul(a1, w2) + b2
	a2 = tf.nn.relu(z2)

	z3 = tf.matmul(a2, w3) + b3
	a3 = tf.nn.relu(z3)

	z4 = tf.matmul(a3, w4) + b4
	a4 = tf.nn.sigmoid(z4)

	return a4

def loss_op(preds, y):
	costs = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
	avg_cost = tf.reduce_mean(costs)
	return avg_cost

def train_NN(train_x, train_y, epoch, lr=.01, show_graph=True):
	tf.reset_default_graph()
	ops.reset_default_graph()
	placeholder_x, placeholder_y = create_placeholders()
	params = create_variables()
	preds = foward_pass(placeholder_x, params)
	avg_cost = loss_op(preds, placeholder_y)
	optimizer = tf.train.AdamOptimizer(lr).minimize(avg_cost)
	initialize = tf.global_variables_initializer()

	costs = []
	batch_costs = []
	with tf.Session() as sess:
		sess.run(initialize)
		for i in range(1, epoch + 1):
			for b in range(len(train_x)):
				_, cur_cost, cur_preds, cur_params = sess.run([optimizer, avg_cost, preds, params], {placeholder_x : train_x[b], placeholder_y : train_y[b]})
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






