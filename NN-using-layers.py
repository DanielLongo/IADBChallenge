import tensorflow as tf
from ReadData import read_data
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import time

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

	def foward_pass(self, x):
		a1 = tf.contrib.layers.fully_connected(x, 100)
		a2 = tf.contrib.layers.fully_connected(a1, 64)
		a3 = tf.contrib.layers.fully_connected(a2, 16)
		a4 = tf.contrib.layers.fully_connected(a3, 4, activation_fn=tf.nn.sigmoid)
		return a4


	def loss_op(self, preds, y):
		costs = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
		avg_cost = tf.reduce_mean(costs)
		return avg_cost

	def build_NN(self):
		ops.reset_default_graph()
		self.create_placeholders()
		self.preds = self.foward_pass(self.placeholder_x)
		self.avg_cost = self.loss_op(self.preds, self.placeholder_y)
		tf.summary.scalar('Cost', self.avg_cost)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.avg_cost)	
		self.initializer = tf.global_variables_initializer()

	def load_data(self):
		self.train_x, self.train_y, self.eval_x, self.eval_y, self.test_x, self.test_y = read_data(self.data_filepath)

	def train_NN(self, epochs, lr, show_graph=True):
		self.epochs = epochs
		self.lr = lr
		nn.build_NN()
		# tf.reset_default_graph()
		self.costs = []
		self.iter_counter = 0
		batch_costs = []
		with tf.Session() as sess:
			sess.run(self.initializer)

			summary_filename = "/train-" + str(int(time.time())) + ""
			self.train_writer = tf.summary.FileWriter(self.logs_path + summary_filename, sess.graph) #creates a summary path for files 
			for i in range(1, self.epochs + 1):
				for b in range(len(self.train_x)):
					cur_batch_x = self.train_x[b]
					cur_batch_y = self.train_y[b]
					cur_feed = {self.placeholder_x : cur_batch_x, self.placeholder_y : cur_batch_y}
					_, cur_cost, preds, summary = sess.run([self.optimizer, self.avg_cost, self.preds, tf.summary.merge_all()], feed_dict=cur_feed)
				
					#tensorboard log			
					self.train_writer.add_summary(summary, self.iter_counter)
					self.train_writer.flush()
					self.iter_counter += 1

					if (i % 10 == 0) and (b == 1):
						print("Cost @", i, cur_cost)
					batch_costs += [cur_cost]

				self.costs += [sum(batch_costs)/len(batch_costs)]

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

	def eval_NN(self):
		preds = tf.argmax(self.preds, 1)
		labels = tf.argmax(self.placeholder_y, 1)
		correct = tf.equal(preds, labels)
		accuracy = tf.reduce_mean(tf.cast(correct, "float"))
		
		train_acc = []
		for b in range(len(self.train_x)):
			cur_batch_x = self.train_x[b]
			cur_batch_y = self.train_y[b]
			cur_feed = {self.placeholder_x : cur_batch_x, self.placeholder_y : cur_batch_y}
			cur_train_acc = accuracy.eval(cur_feed)
			train_acc += [cur_train_acc]
		avg_train_acc = sum(train_acc)/len(train_acc)

		eval_acc = []
		for b in range(len(self.eval_x)):
			cur_batch_x = self.eval_x[b]
			cur_batch_y = self.eval_y[b]
			cur_feed = {self.placeholder_x : cur_batch_x, self.placeholder_y : cur_batch_y}
			cur_eval_acc = accuracy.eval(cur_feed)
			eval_acc += [cur_eval_acc]
		avg_eval_acc = sum(eval_acc)/len(eval_acc)

		print("Average train accuracy:", avg_train_acc*100 , "%")
		print("Average eval accuracy:", avg_eval_acc*100 , "%")

		return {"avg_train_acc" : avg_train_acc, "avg_eval_acc" : avg_eval_acc}

	def compare_runs(self):
		for run in self.runs:
			costs = run["costs"]
			train_acc = run["train_acc"]
			eval_acc = run["eval_acc"]
			lr = run["lr"]
			epochs = run["epochs"]
			label = "epochs: " + str(epochs) + ", lr: " + str(lr)
			self.plot_run(costs, label)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.show()


	def plot_run(self, costs, label):
		epochs = [i for i in range(len(costs))]
		plt.plot(x=epochs, y=costs, label=label)




nn = NN()
nn.load_data()
nn.train_NN(20, .000005, show_graph=False)
print("FInished with A")
nn.train_NN(15, .000005, show_graph=False)
print("FInished with B")
nn.train_NN(10, .000005, show_graph=False)
print("FInished with C")
nn.compare_runs()