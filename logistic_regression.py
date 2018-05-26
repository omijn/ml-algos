from utils import data_transform
import numpy as np

class LogisticRegression():
	def __init__(self):
		pass

	def train(self, X, y):
		X = data_transform.insert_ones(X)		