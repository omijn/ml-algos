# linear regression from scratch
import numpy as np

import data
import metrics

class LinearRegression:
	
	def __init__(self, solve_by='ols', regularized=False):
		self.solve_by = solve_by
		self.regularized = regularized		

	def train(self, features, labels):
		X = data.insert_ones(features)
		y = labels

		if self.solve_by == 'ols':
			self.weights = self.__ols(X, y)
		
		elif self.solve_by == 'gdesc':
			self.weights = self.__gradient_descent(X, y)

	def __ols(self, X, y):
		return np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T), y)

	def __gradient_descent(self, X, y, learning_rate=10e-2, num_iter=1000, stopping_threshold=10e-4):
		# initial weights
		n = X.shape[0]
		weights = np.zeros(X.shape[1])		
		
		for _iter in range(num_iter):
			gradients = (1 / n) * np.matmul(X.T, np.dot(X, weights) - y)	# vectorized gradient calculation

			old_weights = weights
			weights = weights - learning_rate * gradients 	# gradient descent
			weight_difference = np.linalg.norm(weights - old_weights)
			
			if weight_difference <= stopping_threshold:		# break early if we're doing well
				break
		
		return weights

	def predict(self, features):
		X = data.insert_ones(features)
		predictions = np.matmul(X, self.weights)
		return predictions

	def compare_sklearn(self, training_features, training_labels, test_features, test_labels, my_mse, my_r2):
		
		# do linear regression with sklearn
		from sklearn.linear_model import LinearRegression
		skmodel = LinearRegression()		
		skmodel.fit(training_features, training_labels)
		skpreds = skmodel.predict(test_features)		

		# get sklearn mse and r2 score
		skmse = metrics.mean_squared_error(test_labels, skpreds)
		skr2 = metrics.r2_score(test_labels, skpreds)
		
		print("Your model's MSE: {}\nsklearn's MSE: {}".format(my_mse, skmse))
		print()
		print("Your model's R2 score: {}\nsklearn's R2 score: {}".format(my_r2, skr2))


def main():
	training_features, training_labels, test_features, test_labels = data.boston_split(0.87)	

	solve_by = 'gdesc'
	
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	training_features = scaler.fit_transform(training_features)
	test_features = scaler.transform(test_features)

	model = LinearRegression(solve_by=solve_by)
	model.train(training_features, training_labels)
	predictions = model.predict(test_features)

	mse = metrics.mean_squared_error(test_labels, predictions)
	r2_score = metrics.r2_score(test_labels, predictions)

	model.compare_sklearn(training_features, training_labels, test_features, test_labels, mse, r2_score)
	
if __name__ == '__main__':
	main()


