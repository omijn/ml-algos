import data
import metrics
import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

def main():
	X_train, y_train, X_test, y_test = data.boston_split(0.87)

	solve_by = 'gdesc'
	
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	model = LinearRegression(solve_by=solve_by)
	model.train(X_train, y_train)
	predictions = model.predict(X_test)

	mse = metrics.mean_squared_error(y_test, predictions)
	r2_score = metrics.r2_score(y_test, predictions)

	model.compare_sklearn(X_train, y_train, X_test, y_test, mse, r2_score)
	
	# print(X_train[:, 0])
	plt.scatter(X_train[:, 10], y_train, color='red')
	# plt.plot(X_train[:, 1], model.predict(X_train), color='blue')
	plt.title("")
	plt.xlabel("Boston Dataset Features")
	plt.ylabel("Output")
	plt.show()

if __name__ == '__main__':
	main()


