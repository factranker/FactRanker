import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from imblearn.under_sampling import (EditedNearestNeighbours, RandomUnderSampler,
                                     RepeatedEditedNearestNeighbours)

def plot_feature_importance(X, y, names):

    rs = RandomUnderSampler(random_state=55)
    X,y = rs.fit_sample(X,y)
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
               axis=0)
    origindices = np.argsort(importances)[::-1]
    end = min(60, len(X[0]))
    indices = origindices[:end]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(end):
        print("%d. %s (%f)" % (f + 1, names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(end), importances[indices],
         color="black", yerr=std[indices], align="center")
    ordered_names = [names[ii] for ii in indices]
    plt.xticks(range(end), ordered_names[:end] ,rotation="vertical")
    plt.xlim([-1, end])
    plt.ylim([0,0.02])
    plt.tight_layout()
    plt.show()
    return origindices

def get_selector(name, k=50):
	if(name == "PCA"):
		return PCA(n_componenets=k)
	if(name == "gini"):
		forest = ExtraTreesClassifier(n_estimators=250,random_state=785)
		return SelectFromModel(forest, threshold="1.2*mean")
	if(name == "rfecv"):
		return RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
	              scoring='accuracy')



def recursive_elimination(X, y, X_t):
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),scoring='roc_auc')
    X_new = rfecv.fit_transform(X,y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    return X_new, rfecv.transform(X_t)

	
	