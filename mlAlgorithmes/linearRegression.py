

class LinearRegression:

	def __init__(self, lr=0.001, n_iter=1000):
		self.lr = lr
		self.n_iter = n_iter
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		# init parameters
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.n_iter):
			

	def predict(self, X):
		pass

	