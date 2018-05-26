from utils import data_transform
import numpy as np

class LogisticRegression():
	def __init__(self):
		pass

	def train(self, X, y):
		X = data_transform.insert_ones(X)

		self.weights = self.__gradient_descent(X, y)

	def __sigmoid(self, value):
		return 1 / (1 + np.exp(value * -1))

	def __gradient_descent(self, X, y, learning_rate=10e-2, num_iter=3000, stopping_threshold=10e-4):
		n = X.shape[0]
		weights = np.zeros(X.shape[1])

		for _iter in range(num_iter):
			gradients = (1 / n) * np.matmul(X.T, self.__sigmoid(np.dot(X, weights)) - y)	# vectorized gradient calculation

			old_weights = weights
			weights = weights - learning_rate * gradients 	# gradient descent
			weight_difference = np.linalg.norm(weights - old_weights)
			
			if weight_difference <= stopping_threshold:		# break early if we're doing well
				break
		
		return weights

	def predict(self, X):
		X = data_transform.insert_ones(X)
		predictions = (self.__sigmoid(np.dot(X, self.weights)) >= 0.5).astype(int)
		return predictions

	def performance(self, y_true, y_pred):
		from sklearn.metrics import f1_score
		return f1_score(y_true, y_pred)

	def sklearn_performance(self, X_train, X_test, y_train, y_test):
		from sklearn.linear_model import LogisticRegression
		skmodel = LogisticRegression()		
		skmodel.fit(X_train, y_train)
		skpreds = skmodel.predict(X_test)

		from sklearn.metrics import f1_score
		return f1_score(y_test, skpreds)