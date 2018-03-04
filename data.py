import numpy as np
from sklearn.datasets import load_boston

def boston_split(train_proportion=0.8, shuffle=False):
	data = load_boston().data

	dataset_len = data.shape[0]
	
	try:
		assert 0 < train_proportion < 1		
	except Exception as e:
		raise e

	# split training and test data
	train_size = int(train_proportion * dataset_len)
	test_size = dataset_len - train_size

	training_data = data[:train_size]
	test_data = data[-1 * test_size:]	

	# split features and labels
	training_features = training_data[:,0:-1]
	training_labels = training_data[:,-1]

	test_features = test_data[:,0:-1]
	test_labels = test_data[:,-1]

	return training_features, training_labels, test_features, test_labels


def insert_ones(features):
	"""insert bias of 1 before every feature vector"""

	return np.insert(features, 0, np.ones(len(features)), axis=1)