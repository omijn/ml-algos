# linear regression from scratch
import sklearn.linear_model
import sklearn.metrics
import numpy as np
import metrics
import data

class LinearRegression:
	
	def __init__(self, solve_by='ols', regularized=False):
		self.solve_by = 'ols'
		self.regularized = regularized		

	def train(self, features, labels):	
		X = data.insert_ones(features)
		y = labels

		self.weights = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T), y)

	def predict(self, features):
		X = data.insert_ones(features)
		predictions = np.matmul(X, self.weights)
		return predictions

	def compare_sklearn(self, training_features, training_labels, test_features, test_labels, mse):		
		skmodel = sklearn.linear_model.LinearRegression()
		skmodel.fit(training_features, training_labels)
		skpreds = skmodel.predict(test_features)
		skmse = sklearn.metrics.mean_squared_error(test_labels, skpreds)
		print("Your model's MSE: {}\nsklearn's MSE: {}".format(mse, skmse))


def main():	
	training_features, training_labels, test_features, test_labels = data.boston_split(0.87)

	model = LinearRegression()
	model.train(training_features, training_labels)
	predictions = model.predict(test_features)

	mse = metrics.mean_squared_error(test_labels, predictions)	

	model.compare_sklearn(training_features, training_labels, test_features, test_labels, mse)
	
if __name__ == '__main__':
	main()


