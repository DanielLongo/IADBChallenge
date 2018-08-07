import tensorflow as tf
from ReadData import read_data
from ReadData import create_batches
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from random import shuffle
class NN(object):
	def __init__(self):
		self.data_filepath = "./data/train.csv"
		self.logs_path = "./tensorboard"
		self.runs = []
		#each run is a dict {costs:list, train_acc: float, eval_acc:float, lr:float, epochs:int}

	def create_placeholders(self, ):
		self.placeholder_x = tf.placeholder(tf.float32, shape=(None, 140))
		self.placeholder_y = tf.placeholder(tf.float32, shape=(None, 4))

#POSSIBLE FIXES TO GRADS
#print out weights
#print out gradients
#recheck ones 

	def foward_pass(self, x):
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

		# tensorboard log
		with tf.variable_scope("h1", reuse=True):
			weights = tf.get_variable("kernel")
			biases = tf.get_variable("bias")
			tf.summary.histogram("w1", weights)
			tf.summary.histogram("b1", biases)
			# tf.summary.histogram("a1", a1)		
		with tf.variable_scope("h2", reuse=True):
			weights = tf.get_variable("kernel")
			biases = tf.get_variable("bias")
			tf.summary.histogram("w2", weights)
			tf.summary.histogram("b2", biases)
			# tf.summary.histogram("a2", a2)		
		with tf.variable_scope("h3", reuse=True):
			weights = tf.get_variable("kernel")
			biases = tf.get_variable("bias")
			tf.summary.histogram("w3", weights)
			tf.summary.histogram("b3", biases)
			# tf.summary.histogram("a3", a3)		
		with tf.variable_scope("h4", reuse=True):
			weights = tf.get_variable("kernel")
			biases = tf.get_variable("bias")
			tf.summary.histogram("w4", weights)
			tf.summary.histogram("b4", biases)
			# tf.summary.histogram("a4", a4)		

		return a4


	def loss_op(self, preds, y):
		costs = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
		avg_cost = tf.reduce_mean(costs)
		return avg_cost

	def build_NN(self, lr):
		self.initializer = tf.global_variables_initializer()
		self.local_initializer = tf.local_variables_initializer()
		with (self.graph).as_default():
			self.create_placeholders()

			self.preds = self.foward_pass(self.placeholder_x)
			self.avg_cost = self.loss_op(self.preds, self.placeholder_y)
			self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.avg_cost)	

			tf.summary.scalar('Cost', self.avg_cost)
			self.summary_op = tf.summary.merge_all()


			self.true_preds = tf.argmax(self.preds, 1)
			self.true_labels = tf.argmax(self.placeholder_y, 1)
			self.correct = tf.equal(self.true_preds, self.true_labels)
			# self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"))
			self.accuracy, _ = tf.metrics.accuracy(self.true_labels, self.true_preds)
			self.precision, _ = tf.metrics.precision(self.true_labels, self.true_preds)
			self.recall, _ = tf.metrics.recall(self.true_labels, self.true_preds)

	def load_data(self):
		self.train_x, self.train_y, self.eval_x, self.eval_y, self.test_x, self.test_y = read_data(self.data_filepath)

	def train_NN(self, epochs, lr, batch_size=16, show_graph=True, print_costs=False):
		iter_counter = 0 #counts iters for tensorboard
		batch_costs = [] #costs across batches
		costs = [] #avg batch costs
		num_batches = int(self.train_x.shape[0] / batch_size) + 1
		self.graph = tf.Graph()
		nn.build_NN(lr)
		with tf.Session(graph=self.graph) as sess:
			sess.run(self.initializer)
			sess.run(self.local_initializer)
			# sess.run(tf.local_variables_initializer())
			# sess.run(tf.global_variables_initializer())
			summary_filename = "/train-" + str(int(time.time())) + ""
			self.train_writer = tf.summary.FileWriter(self.logs_path + summary_filename, sess.graph) #creates a summary path for files 
			
			for i in range(epochs):
				item_order = list(range(self.train_x.shape[0]))
				shuffle(item_order)
				for b in range(num_batches):
					if (b == num_batches - 1) and (b*batch_size != self.train_x.shape[0]):
						item_indexes = item_order[b*batch_size:]
					else:
						item_indexes = item_order[b*batch_size:(b+1)*batch_size]

					cur_batch_x = self.train_x[item_indexes]
					cur_batch_y = self.train_y[item_indexes]

					cur_feed = {self.placeholder_x : cur_batch_x, self.placeholder_y : cur_batch_y}
					_, cur_cost, preds, summary = sess.run([self.optimizer, self.avg_cost, self.preds, self.summary_op], feed_dict=cur_feed)
				
					#tensorboard log			
					self.train_writer.add_summary(summary, iter_counter) 
					self.train_writer.flush()
					iter_counter += 1

					if ((i + 1) % 10 == 0) and (b == 1) and print_costs:
						print("Cost @", i + 1, cur_cost)

					batch_costs += [cur_cost]

				costs += [sum(batch_costs)/len(batch_costs)]
			print(sess.run(self.precision, cur_feed))

			if show_graph:
				plot_costs(costs)

			score = self.eval_NN()
			train_acc = score["avg_train_acc"]
			eval_acc = score["avg_eval_acc"]
			run_log = {"costs": costs, "train_acc": train_acc, "eval_acc": eval_acc, "lr" : lr, "epochs" : epochs}
			self.runs += [run_log]
			
			return sess

	def plot_costs(costs):
		epochs = list(range(len(costs)))
		plt.plot(x, y)
		plt.ylabel("Cost")
		plt.xlabel("Epoch")
		plt.show()


	def predict(self, x):
		preds = tf.argmax(se)

	def eval_given_examples(self, examples_x, examples_y, batch_size=16):
		batches_x, batches_y = create_batches(self.train_x, self.train_y, batch_size)
		acc = []
		precision = []
		recall = []
		# with tf.Session(graph=self.graph) as sess: 
		# 	for b in range(len(batches_x)):
		# 		cur_batch_x = batches_x[b]
		# 		cur_batch_y = batches_y[b]
		# 		print("SHAPE OF KFNDSFKL", np.shape(cur_batch_x))
		# 		cur_feed = {self.placeholder_x : cur_batch_x, self.placeholder_y : cur_batch_y}
		# 		cur_acc, cur_precision, cur_recall = sess.run([self.accuracy, self.precision, self.recall])
		# 		cur_precision = self.precision.eval(cur_feed)
		# 		cur_recall = self.recall.eval(cur_feed)
		# 		cur_acc = self.accuracy.eval(cur_feed)

		# 		acc += [cur_acc]
		# 		precision += [cur_precision]
		# 		recall += [cur_recall]

		# avg_acc = sum(acc)/len(acc)
		# avg_precision = sum(precision)/len(precision)
		# avg_recall = sum(recall)/len(recall)
		for b in range(len(batches_x)):
			cur_batch_x = batches_x[b]
			cur_batch_y = batches_y[b]
			cur_feed = {self.placeholder_x : cur_batch_x, self.placeholder_y : cur_batch_y}
			cur_precision = self.precision.eval(cur_feed)
			cur_recall = self.recall.eval(cur_feed)
			cur_acc = self.accuracy.eval(cur_feed)

			acc += [cur_acc]
			precision += [cur_precision]
			recall += [cur_recall]

		avg_acc = sum(acc)/len(acc)
		avg_precision = sum(precision)/len(precision)
		avg_recall = sum(recall)/len(recall)
		return avg_acc, avg_precision, avg_recall


	def eval_NN(self):
		train_acc, train_precision, train_recall = self.eval_given_examples(self.train_x, self.train_y)
		print("train_acc", train_acc)
		print("train_precision", train_precision)
		print("train_recall", train_recall)

		eval_acc, eval_precision, eval_recall = self.eval_given_examples(self.eval_x, self.eval_y)
		print("eval_acc", eval_acc)
		print("eval_precision", eval_precision)
		print("eval_recall", eval_recall)

		# print("Average train accuracy:", avg_train_acc*100 , "%")
		# print("Train precision / recall", train_precision, "/", train_recall)
		# print("train f1 score", 2 * ((train_precision * train_recall) / (train_precision + train_recall)))
		# print("Average eval accuracy:", avg_eval_acc*100 , "%")
		# print("Train precision / recall", avg_eval_precision, "/", avg_eval_recall)
		# print("train f1 score", 2 * ((avg_eval_precision * avg_eval_recall) / (avg_eval_precision + avg_eval_recall)))

		return {"avg_train_acc" : train_acc, "avg_eval_acc" : eval_acc}

	def compare_runs(self, show_graph=True):
		for run in self.runs:
			costs = run["costs"]
			train_acc = run["train_acc"]
			eval_acc = run["eval_acc"]
			lr = run["lr"]
			epochs = run["epochs"]
			label = "epochs: " + str(epochs) + ", lr: " + str(lr)
			print("Run Stats:")
			print("learning rate:", lr, "Num of Epochs", epochs)
			print("Train Acc", train_acc, "Eval Acc", eval_acc)
			self.plot_run(costs, label)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		if show_graph:
			plt.show()



	def plot_run(self, costs, label):
		epochs = [i for i in range(len(costs))]
		plt.plot(x=epochs, y=costs, label=label)



nn = NN()
nn.load_data()
nn.train_NN(10, .001, show_graph=False, print_costs=True)
# nn.train_NN(10, .00025, show_graph=False)
# print("Finished 2")
# nn.train_NN(100, .01, show_graph=False)
# print("Finished 3")
# # nn.compare_runs(show_graph=False)
# print("FInished with All")