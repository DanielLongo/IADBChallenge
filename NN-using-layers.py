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
		self.epochs = None
		self.lr = None
		self.costs = []
		self.data_filepath = "./data/train.csv"
		self.logs_path = "./tensorboard"
		self.iter_counter = 0 #counts iterations of model for summary writer
		self.runs = []
		#each run is a dict {costs:list, train_acc: float, eval_acc:float, lr:float, epochs:int}

	def create_placeholders(self, ):
		self.placeholder_x = tf.placeholder(tf.float32, shape=(None, 140))
		self.placeholder_y = tf.placeholder(tf.float32, shape=(None, 4))


	def log_layer(self, layer_name, w=True, b=True, a=False, x=False):
		#TODO: get w, a and x to work

		layer_number = layer_name[-1]
		with tf.variable_scope(layer_name, reuse=True):
			if w:
				weights = tf.get_variable("kernel")
				summary_name = "w" + layer_number
				tf.summary.histogram(summary_name, w)
			if b:
				bias  = tf.get_variable("bias")
				summary_name = "b" + layer_number
				tf.summary.histogram(summary_name, bias)
			if a != False:
				summary_name = "a" + layer_number
				tf.summary.histogram(a, summary_name)
			if x != False:
				summary_name = "x" + layer_number
				tf.summary.histogram(x, summary_name)

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

	def build_NN(self):
		ops.reset_default_graph()
		tf.reset_default_graph()
		self.initializer = tf.global_variables_initializer()
		self.local_initializer = tf.local_variables_initializer()
		self.create_placeholders()
		self.preds = self.foward_pass(self.placeholder_x)
		self.avg_cost = self.loss_op(self.preds, self.placeholder_y)
		tf.summary.scalar('Cost', self.avg_cost)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.avg_cost)	
		round_preds = tf.argmax(self.preds, 1)
		round_lables = tf.argmax(self.placeholder_y, 1)
		self.precision, _ = tf.metrics.precision(round_lables, round_preds)
		self.recall, _ = tf.metrics.recall(round_lables, round_preds)
		self.summary_op = tf.summary.merge_all()


	def load_data(self):
		self.train_x, self.train_y, self.eval_x, self.eval_y, self.test_x, self.test_y = read_data(self.data_filepath)

	def train_NN(self, epochs, lr, show_graph=True):
		self.epochs = epochs
		self.lr = lr
		nn.build_NN()
		self.costs = []
		self.iter_counter = 0
		batch_costs = []
		BATCH_SIZE = 16
		num_batches = int(self.train_x.shape[0] / BATCH_SIZE) + 1
		with tf.Session() as sess:
			# sess.run(self.initializer)
			sess.run(tf.local_variables_initializer())
			# tf.global_variables_initializer().run()
			sess.run(tf.global_variables_initializer())
			# tf.initialize_all_variables().run()
			summary_filename = "/train-" + str(int(time.time())) + ""
			self.train_writer = tf.summary.FileWriter(self.logs_path + summary_filename, sess.graph) #creates a summary path for files 
			
			for i in range(1, self.epochs + 1):
				# train_batch = list(zip(self.train_x, self.train_y))
				# shuffle(train_batch)
				# self.train_x, self.train_y = zip(*train_batch)				
				item_order = list(range(self.train_x.shape[0]))
				# shuffle(item_order)
				# for b in range(len(self.train_x)):
				for b in range(num_batches):
					if (b == num_batches - 1) and (b*BATCH_SIZE != self.train_x.shape[0]):
						item_indexes = item_order[b*BATCH_SIZE:]
					else:
						item_indexes = item_order[b*BATCH_SIZE:(b+1)*BATCH_SIZE]

					cur_batch_x = self.train_x[item_indexes]
					cur_batch_y = self.train_y[item_indexes]


					cur_feed = {self.placeholder_x : cur_batch_x, self.placeholder_y : cur_batch_y}
					_, cur_cost, preds, summary = sess.run([self.optimizer, self.avg_cost, self.preds, self.summary_op], feed_dict=cur_feed)
					# _, cur_cost, preds = sess.run([self.optimizer, self.avg_cost, self.preds], feed_dict=cur_feed)
				
					#tensorboard log			
					self.train_writer.add_summary(summary, self.iter_counter) 
					self.train_writer.flush()
					self.iter_counter += 1

					if (i % 10 == 0) and (b == 1):
						print("Cost @", i, cur_cost)
					batch_costs += [cur_cost]

				self.costs += [sum(batch_costs)/len(batch_costs)]
			print(sess.run(self.precision, cur_feed))
			if show_graph:
				y = self.costs
				x = [i for i in range(len(y))]

				plt.plot(x, y)
				plt.ylabel("Cost")
				plt.xlabel("Epoch")
				plt.show()

			score = self.eval_NN()
			train_acc = score["avg_train_acc"]
			eval_acc = score["avg_eval_acc"]
			run_log = {"costs": self.costs, "train_acc": train_acc, "eval_acc": eval_acc, "lr" : self.lr, "epochs" : self.epochs}
			self.runs += [run_log]
			return sess

	def predict(self, x):
		preds = tf.argmax(se)

	def eval_NN(self):
		preds = tf.argmax(self.preds, 1)
		labels = tf.argmax(self.placeholder_y, 1)
		correct = tf.equal(preds, labels)
		accuracy = tf.reduce_mean(tf.cast(correct, "float"))
		train_acc = []
		train_precision = []
		train_recall = []

		batches_x, batches_y = create_batches(self.train_x, self.train_y, 16)
		for b in range(len(batches_x)):
			cur_batch_x = batches_x[b]
			cur_batch_y = batches_y[b]
			cur_feed = {self.placeholder_x : cur_batch_x, self.placeholder_y : cur_batch_y}
			cur_precision = self.precision.eval(cur_feed)
			cur_recall = self.recall.eval(cur_feed)
			cur_train_acc = accuracy.eval(cur_feed)

			train_acc += [cur_train_acc]
			train_precision += [cur_precision]
			train_recall += [cur_recall]

		avg_train_acc = sum(train_acc)/len(train_acc)
		train_precision = sum(train_precision)/len(train_precision)
		train_recall = sum(train_recall)/len(train_recall)

		eval_acc = []
		eval_precision = []
		eval_recall = []
		batches_x, batches_y = create_batches(self.eval_x, self.eval_y, 16)
		for b in range(len(batches_x)):
			cur_batch_x = batches_x[b]
			cur_batch_y = batches_y[b]
			cur_feed = {self.placeholder_x : cur_batch_x, self.placeholder_y : cur_batch_y}
			cur_eval_acc = accuracy.eval(cur_feed)
			cur_precision = self.precision.eval(cur_feed)
			cur_recall = self.recall.eval(cur_feed)

			eval_acc += [cur_eval_acc]
			eval_precision += [cur_precision]
			eval_recall += [cur_recall]

		avg_eval_acc = sum(eval_acc)/len(eval_acc)
		avg_eval_precision = sum(eval_precision)/len(eval_precision)
		avg_eval_recall = sum(eval_recall)/len(eval_recall)

		print("Average train accuracy:", avg_train_acc*100 , "%")
		print("Train precision / recall", train_precision, "/", train_recall)
		print("train f1 score", 2 * ((train_precision * train_recall) / (train_precision + train_recall)))
		print("Average eval accuracy:", avg_eval_acc*100 , "%")
		print("Train precision / recall", avg_eval_precision, "/", avg_eval_recall)
		print("train f1 score", 2 * ((avg_eval_precision * avg_eval_recall) / (avg_eval_precision + avg_eval_recall)))

		return {"avg_train_acc" : avg_train_acc, "avg_eval_acc" : avg_eval_acc}

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
nn.train_NN(10, .001, show_graph=False)
# nn.train_NN(10, .00025, show_graph=False)
# print("Finished 2")
# nn.train_NN(100, .01, show_graph=False)
# print("Finished 3")
# # nn.compare_runs(show_graph=False)
# print("FInished with All")