import numpy as np

def mean_squared_error(y_true, y_pred):
	return np.sum(np.square(y_true - y_pred)) / len(y_true)