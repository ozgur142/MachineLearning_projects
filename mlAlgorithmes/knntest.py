import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


print(X_train.shape, y_train.shape)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', s=20)
plt.show()

from knn import knn

model = knn(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

