import numpy as np

def mean_squared_error(y_true, y_pred):
	return np.sum(np.square(y_true - y_pred)) / len(y_true)

def r2_score(y_true, y_pred):
	from sklearn.metrics import r2_score
	return r2_score(y_true, y_pred)