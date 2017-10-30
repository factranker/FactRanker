import ast
import numpy as np
import scipy
import pickle
import os
import json
import pprint
import itertools
import pandas as pd
pp = pprint.PrettyPrinter(indent=4)

from collections import Counter

from time import time
from collections import defaultdict

import matplotlib.pyplot as plt
import sklearn
#Sk-learn metrics
from sklearn import metrics
# accuracy_score,average_precision_score,f1_score,log_loss,precision_score,recall_score,roc_auc_score,roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import decomposition
from sklearn import ensemble

from EnsembleEstimator import EnsembleEstimator
from EnsembleEstimator import ClassifierComb
import feature_importance as fi
from imblearn.under_sampling import (EditedNearestNeighbours, RandomUnderSampler,
                                     RepeatedEditedNearestNeighbours)
from utils import rank_scorers

# Colors for plotting

import seaborn as sns
sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

def load_dataset():
	try:
		raise Exception()
		dataset = pickle.load(open("dataset/both_dataset.pkl", "rb"))
	except:
		# filename = "dataset/all471+tfidf+buster.json"
		filename = "both_datasets.json"
		
		with open(filename) as f:
			dataset = json.load(f)


		data = dataset['data']
		# print(data[55])

		print(len(data), len(data[0]))
		data = pd.lib.to_object_array(data)
		# data = np.asarray(data)
		print(type(data))

		dataset['data'] = data

		pickle.dump(dataset,open("dataset/both_dataset.pkl", "wb"))

	OFFSET = 2
	data = dataset['data']
	# print(data[55])
	print(type(data), type(data[0]))
	print(len(data), len(data[0]))
	X = data[:, OFFSET:-1]
	X_meta = data[:, :OFFSET]
	X = X.astype('float32')	
	Y = data[:, -1].astype('int')
	attr = dataset['attributes']
	# for i in range(len(attr)):
	# 	if(type(attr[i]) == list):
	# 		print(attr)
	# 		attr[i] = attr[i][0]
	# print(attr)

	# attr = np.asarray(attr)

	print("Dataset Loaded: Shape = ", X.shape, Y.shape)
	# print(X[0], Y[0])


	return attr[OFFSET:], X, Y, X_meta


def plot_data(X, Y):

	pca = decomposition.PCA(n_components=2)
	X_2 = pca.fit_transform(X)
	plt.scatter(X_2[Y==0, 0], X_2[Y==0, 1], label="Class #0", alpha=0.5, 
		edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
	plt.scatter(X_2[Y==1, 0], X_2[Y==1, 1], label="Class #1", alpha=0.5, 
		edgecolor=almost_black, facecolor="orange", linewidth=0.15)

	plt.show()



def train(X, Y):
	# plot_data(X, Y)
	sampler = RandomUnderSampler(random_state=487)

	X_s, Y_s = sampler.fit_sample(X, Y)
	clf = ensemble.RandomForestClassifier(n_estimators=20)

	clfs = [("SVM_RBF", svm.SVC(kernel="rbf", probability=True)),
		 	("SVM_Lin", svm.SVC(kernel="linear", probability=True)),
		 	("RF", ensemble.RandomForestClassifier(n_estimators=20)),
		 	("AB", ensemble.AdaBoostClassifier(n_estimators=10)),
		 	("GradBoost", ensemble.GradientBoostingClassifier()),
		 	("Logistic", sklearn.linear_model.LogisticRegression()),
		 	("E_SVML", EnsembleEstimator(svm.SVC, {"kernel":"linear","probability":True})),
		 	("E_RF", EnsembleEstimator(ensemble.RandomForestClassifier, {"n_estimators":20}))
		 	]

	clfs_f = []
	for i in range(6):
		name, clf = clfs[i]
		clf.fit(X_s, Y_s)
		clfs_f.append((clf, name))

	for i in range(7,8):
		name, clf = clfs[i]
		clf.fit(X, Y)
		clfs_f.append((clf, name))

	# clf = svm.SVC(kernel="rbf", probability=True)
	# clf = svm.SVC(kernel="linear", probability=True)
	# clf = sklearn.ensemble.RandomForestClassifier(n_estimators=20)
	# clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=10)
	# clf = EnsembleEstimator(svm.SVC, {"kernel":"linear","probability":True})
	# clf = EnsembleEstimator(sklearn.ensemble.RandomForestClassifier, {"kernel":"linear","probability":True})

	# clf.fit(X, Y)

	return clfs_f

def test_random(X, Y, clf):
	from imblearn.ensemble import EasyEnsemble
	ees = EasyEnsemble(n_subsets=10)

	X, Y = ees.fit_sample(X, Y)
	ps = []
	rs = []
	for i in range(len(X)):
		Xi, Yi = X[i], Y[i]
		y_hati = clf.predict(Xi)

		pi = metrics.precision_score(Yi, y_hati)
		ri = metrics.recall_score(Yi, y_hati)

		ps.append(pi)
		rs.append(ri)
	ps = np.asarray(ps)
	rs = np.asarray(rs)

	mean_p = np.mean(ps)
	mean_r = np.mean(rs)

	std_p = np.std(ps)
	std_r = np.std(rs)

	return mean_p, 2*std_p, mean_r, 2*std_r





def test(X, Y, clf):
	y_hat = clf.predict(X)
	print(metrics.classification_report(Y, y_hat))

	print("Precision = %.2f+/-%.2f, Recall = %.2f+/-%.2f\n"%(test_random(X, Y, clf)))


def evaluate(X_test, y_test, meta, clf, name, prefix):

    y_hat = clf.predict(X_test)
    report = metrics.classification_report(y_test, y_hat)


    print(report)
    print("Precision = %.2f+/-%.2f, Recall = %.2f+/-%.2f\n"%(test_random(X_test, y_test, clf)))
    
    try:
        y_prob = clf.predict_proba(X_test)[:,1]
    except:
        pass

    ks = [10,20,30,50,80,100,500,1000]

    # print(list(zip(meta, y_prob)))
    result_file = open("results/paper_"+prefix+name, "w")
    for i in range(len(meta)):
    	print(meta[i], y_test[i], y_prob[i], y_hat[i], file=result_file)


    allscores = rank_scorers.all_score(y_test, y_prob, ks)
        
    for i,k in enumerate(ks):
        print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t")


    allscores = rank_scorers.all_score(y_test, np.round(y_prob, 2), ks)
        
    for i,k in enumerate(ks):
        print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t")

    return metrics.precision_score(y_test, y_hat), metrics.recall_score(y_test, y_hat), metrics.f1_score(y_test, y_hat)


class Dummy:
	def __init__(self, threshold):
		self.threshold = threshold
	def predict(self, X):
		return X > self.threshold
	def predict_proba(self, X):
		X = X.reshape(-1,1)
		return np.hstack((1-X, X))
		

def main():

	attr, X, Y, X_meta = load_dataset()
	print(len(X))

	'''
	ENT_TYPE = range(0,12)
	LIWC = range(12,16)
	TOPIC = range(16,36)
	SENT_SIM = range(36,37)
	BIGRAM = range(37,97)
	DEP = range(97,103)
	EMB = range(103,403)
	POS = range(403,448)
	SENTIMENT = range(448,452)
	SUBJ = range(452,455)
	VERBCAT = range(455,460) 
	PAST_GROUP = range(460,461)
	BUSTER_SCORE = range(461,462)
	ALL = range(0,461)
	TF_IDF = range(462, 2033)

	selected_features = [i for i in itertools.chain(ALL)]
	selected_features = np.asarray(selected_features)
	X = X[:, selected_features]
	'''

	# pp.pprint(attr)
	trainX, testX, trainY, testY, trainMeta, testMeta = sklearn.model_selection.train_test_split(X, Y, X_meta, test_size=0.3, random_state=43, stratify=Y)
	print (trainX.shape, np.sum(testY),testX.shape)
	

	# import feature_importance as fi

	# trainX, testX = fi.recursive_elimination(trainX, trainY, testX)

	# pca = decomposition.PCA(n_components=200)
	# trainX = pca.fit_transform(trainX)
	# testX = pca.transform(testX)

	
	# fi.plot_feature_importance(X, Y, attr[selected_features])
	# imp_ind = fi.plot_feature_importance(trainX, trainY, attr[selected_features])
	# TOP = 150
	# top100 = imp_ind[:TOP]
	# trainX, testX = trainX[:, top100], testX[:, top100]



	scores = []
	for _ in range(10):

		clfs = train(trainX, trainY)

		rbf, _ = clfs[0]
		enrf, _ = clfs[-1]

		clf = ClassifierComb([rbf, enrf])
		p, r, fscore = evaluate(testX, testY, testMeta, clf, "final", "")

		scores.append([p, r, fscore])


	print(np.mean(scores, axis=0), np.std(scores, axis=0))
	# ths = [0.4, 0.45, 0.499, 0.5, 0.55, 0.6, 0.7]
	# for th in ths:
	# 	evaluate(testX, testY, Dummy(th))

	# clfs = train(trainX, trainY)
	# for clf, name in clfs:
	# 	print(name)
	# 	evaluate(testX, testY, testMeta, clf, name, "all_")
	# for i, atr in enumerate(attr[selected_features]):
	# 	print (i, "\t", atr)



main()
