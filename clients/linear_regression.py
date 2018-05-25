from utils import datasets
import metrics
from linear_regression import LinearRegression
import numpy as np

X_train, y_train, X_test, y_test = datasets.boston_split(0.87)

solve_by = 'gdesc'	# the other option is 'ols'

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