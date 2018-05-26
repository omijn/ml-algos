import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from utils import datasets, data_transform
from logistic_regression import LogisticRegression

# dataset info: https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names
data = pd.read_csv("./datasets/credit_approval.txt")

# encode all categorical data as integers. Missing data ('?') gets converted to '0'.
le = LabelEncoder()
data.iloc[:, 0] = le.fit_transform(data.iloc[:, 0])
data.iloc[:, 3] = le.fit_transform(data.iloc[:, 3])
data.iloc[:, 4] = le.fit_transform(data.iloc[:, 4])
data.iloc[:, 5] = le.fit_transform(data.iloc[:, 5])
data.iloc[:, 6] = le.fit_transform(data.iloc[:, 6])
data.iloc[:, 8] = le.fit_transform(data.iloc[:, 8])
data.iloc[:, 9] = le.fit_transform(data.iloc[:, 9])
data.iloc[:, 11] = le.fit_transform(data.iloc[:, 11])
data.iloc[:, 12] = le.fit_transform(data.iloc[:, 12])

# fill in missing values (categorical data): convert all '0's to most frequent value in column
categorical_imputer = Imputer(missing_values=0, strategy='most_frequent')
data.iloc[:, 0] = categorical_imputer.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))
data.iloc[:, 3:7] = categorical_imputer.fit_transform(data.iloc[:, 3:7])

# for numerical columns, change all missing values from '?' to 'NaN'
data = data.replace(to_replace='?', value='NaN')	# format dataset for sklearn imputer to understand

# fill in missing values (numerical data): convert all 'NaN's to mean of column
continuous_imputer = Imputer(missing_values='NaN', strategy='mean')
data.iloc[:, 1] = continuous_imputer.fit_transform(data.iloc[:, 1].values.reshape(-1, 1))
data.iloc[:, 13] = continuous_imputer.fit_transform(data.iloc[:, 13].values.reshape(-1, 1))

# split into features and labels
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# encode + and - as 1 and 0
y = le.fit_transform(y)		# encode: (+, 0), (-, 1)
y = (y == 0).astype(int)	# swap encoding: (+, 1), (-, 0)

# binarize categorical variables
ohe = OneHotEncoder(categorical_features=[0, 3, 4, 5, 6, 8, 9, 11, 12])
X = ohe.fit_transform(X).toarray()

# avoid "dummy variable trap" by removing columns that can be inferred after one hot encoding
X = np.delete(X, [0, 7, 10, 24, 34, 36, 39, 41], axis=1)

# split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)