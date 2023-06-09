import numpy as np
from collections import Counter 

def euclidean_distance(x1, x2):
	return np.sqrt(np.sum((x1-x2)**2))

class knn:
	def __init__(self, k=3):
		self.k = k

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

	def predict(self, X):
		predicted_labels = []
		for x in X:
			distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
			#get k nearest samples, labels
			k_indices = np.argsort(distances)[:self.k]
			k_nearest_labels = [self.y_train[i] for i in k_indices]
			#majority vote, most common label
			most_common = Counter(k_nearest_labels).most_common(1)

			predicted_labels.append(most_common[0][0])

		return np.array(predicted_labels)




