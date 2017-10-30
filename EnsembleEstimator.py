from imblearn.ensemble import EasyEnsemble
import numpy as np
from collections import Counter

class EnsembleEstimator:

	def __init__(self, estimator, kargs, n_subsets=10):

		self.estimator = estimator
		self.n_subsets = n_subsets
		self.kargs = kargs


	def fit(self, X, y):

		ees = EasyEnsemble(n_subsets=self.n_subsets)
		X_res, y_res = ees.fit_sample(X,y)

		self.clf = []
		for i in range(len(X_res)):
			print(Counter(y_res[i]))
			clfi = self.estimator()
			clfi.set_params(**self.kargs)
			clfi.fit(X_res[i], y_res[i])
			self.clf.append(clfi)


	def predict_proba(self, X):
		y_proba = []
		for clfi in self.clf:
			y_probai = clfi.predict_proba(X)[:,-1]
			y_proba.append(y_probai)

		y_proba = np.asarray(y_proba)

		y_proba_mean = np.mean(y_proba, axis=0)


		y_proba_mean = y_proba_mean.reshape(-1,1)
		return np.hstack((1 - y_proba_mean, y_proba_mean))

		# return y_proba_mean

	def predict(self, X):

		y_hat = []
		for clfi in self.clf:
			y_hati = clfi.predict(X)
			y_hat.append(y_hati)

		y_hat_mean = np.mean(y_hat, axis=0)

		return y_hat_mean > 0.5


			

class ClassifierComb:

	def __init__(self, clfs):
		self.clfs = clfs

	def predict_proba(self, X):
		y_proba = []
		for clfi in self.clfs:
			y_probai = clfi.predict_proba(X)[:,-1]
			y_proba.append(y_probai)

		y_proba = np.asarray(y_proba)

		# y_proba_mean = np.average(y_proba, axis=0, weights=[0.01,0.99])

		y_proba_mean = np.mean(y_proba, axis=0)
		y_proba_mean = y_proba_mean.reshape(-1,1)
		return np.hstack((1 - y_proba_mean, y_proba_mean))

	def predict(self, X):

		y_hat = []
		for clfi in self.clfs:
			y_hati = clfi.predict(X)
			y_hat.append(y_hati)

		y_hat_mean = np.mean(y_hat, axis=0)

		return y_hat_mean >= 1
