import pandas as pd
import numpy as np
import math

def read_data(filepath, examples_per_batch=16, train_ratio=.8, eval_ratio=.1, test_ratio=.1):
	assert((train_ratio + eval_ratio + test_ratio) == 1), "Invalid ratios of train, eval, and test data"
	df = pd.read_csv(filepath)
	df = df.fillna(0.0) #fills mising values
	df = df.sample(frac=1) #shuffles all rows
	mapping = {"no" : -1, "yes": 1}
	df = df.replace({"dependency":mapping, "edjefe":mapping, "edjefa":mapping})
	y = df["Target"]
	y = pd.get_dummies(y, "Target") #uses one-hot encoding vector
	df = df.drop(["Id", "Target", "idhogar"], axis=1) 
	df = normalize_df(df)
	#TODO: Normalize non binary x columns
	#steps:
	#take average of column
	#subtract average from each value
	#find sd from column
	#divide by sd
	x = df

	
	m = len(x)
	num_examples_train = math.ceil(m * train_ratio)
	num_examples_eval = int(m * eval_ratio)
	num_examples_test = int(m * test_ratio)
	
	assert((num_examples_train + num_examples_eval + num_examples_test) <= m), "Invalid data split values"

	train_x, train_y = x[:num_examples_train], y[:num_examples_train]
	eval_x, eval_y = x[num_examples_train: num_examples_train + num_examples_eval], y[num_examples_train: num_examples_train + num_examples_eval]
	test_x, test_y = x[m - num_examples_test:], y[m - num_examples_test:]

	assert((len(train_x) + len(eval_x) + len(test_x)) <= m), "Invalid data split"

	print("m:", m)
	print("Number of train examples:", len(train_x))
	print("Number of eval examples:", len(eval_x))
	print("Number of test examples:", len(test_x))

	train_x, train_y = create_batches(train_x, examples_per_batch), create_batches(train_y, examples_per_batch)
	eval_x, eval_y = create_batches(eval_x, examples_per_batch), create_batches(eval_y, examples_per_batch)
	test_x, test_y = create_batches(test_x, examples_per_batch), create_batches(test_y, examples_per_batch)

	return train_x, train_y, eval_x, eval_y, test_x, test_y

def check_feature(array):
	#returns True if feature should be normalized
	#returns False if feature is a class
	num_zeros = (array == 0).sum()
	num_ones = (array == 1).sum()
	sum_of_binary_featues = num_ones + num_zeros
	# print("Sum in check_feature below and then length of array")
	# print("feature", sum_of_binary_featues)
	# print("array length", len(array))
	if (num_ones + num_zeros) == len(array):
		return False
	return True

def normalize_array(array):
	array = array.astype(float) #converts any extraneous strings to floats
	mean = np.sum(array)/len(array)
	array = array - mean
	std = np.std(array)
	normalized_array = array - std
	return normalized_array

def normalize_df(df):
	for feature in df:
		cur_array = df[feature]
		if check_feature(cur_array) == True:
			# print("Normalized feature", feature)
			df[feature] = normalize_array(cur_array)
			continue
		# print("Did not normalized feature", feature)
	
	return df

def create_batches(examples, examples_per_batch):
	#creates batches exclusively
	batches = []
	for i in range(0, len(examples) - examples_per_batch, examples_per_batch):
		start = i
		end = start + examples_per_batch
		batches += [examples[start:end].values]

	return batches



# train_x, train_y, eval_x, eval_y, test_x, test_y = read_data("./data/train.csv")

# print("train_x shape:", np.shape(train_x))
# print("train_y shape:", np.shape(train_y))
# print("eval_x shape:", np.shape(eval_x))
# print("eval_y shape:", np.shape(eval_y))
# print("test_x shape:", np.shape(test_x))
# print("test_y shape:", np.shape(test_y))

