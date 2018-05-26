import numpy as np
from collections import Counter

def insert_ones(features):
	"""insert bias of 1 before every feature vector"""

	return np.insert(features, 0, np.ones(len(features)), axis=1)

def categorical_frequency_imputer(missing_value, pandas_dataframe):
	c = Counter(pandas_dataframe)
	most_common = c.most_common(1)[0][0]
	return pandas_dataframe.replace(to_replace=missing_value, value=most_common)

