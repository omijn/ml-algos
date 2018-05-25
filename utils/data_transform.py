import numpy as np

def insert_ones(features):
	"""insert bias of 1 before every feature vector"""

	return np.insert(features, 0, np.ones(len(features)), axis=1)