#from sklearn.neural_network import MLPClassifier
import ast
import numpy as np
import scipy
import pickle
import os

from collections import Counter

from time import time
from collections import defaultdict
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


#from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


from imblearn.under_sampling import (EditedNearestNeighbours, RandomUnderSampler,
                                     RepeatedEditedNearestNeighbours)
from imblearn.ensemble import EasyEnsemble
from imblearn.pipeline import Pipeline as im_Pipeline

import rank_scorers
import sampler
import feature_importance
import useClaimBuster
import dataset_utils


basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"
workingdir = basepath + "/output_all"

import seaborn as sns
sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()

names = [
"KNN", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis", 
         "MLP","ExtraTreesClassifier"]


classifiers = [
    KNeighborsClassifier(weights='distance', n_neighbors=121),
    SVC(kernel="linear", C=1, probability=True),
    SVC(gamma=2, C=1, probability=True),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto"),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,75,50,25,15), max_iter=10000, random_state=1),
    ExtraTreesClassifier(n_estimators=100,random_state=97)
    ]

SVCs = [
    SVC(kernel="linear",C=1, probability=True),
    SVC(kernel="linear",C=2, probability=True),
    SVC(kernel="linear",C=3, probability=True),
    SVC(kernel="linear",C=4, probability=True),
    SVC(kernel="linear",C=5, probability=True),
    SVC(kernel="linear",C=6, probability=True),
    SVC(kernel="linear",C=7, probability=True),
]
svc_names = [
    'svc1','svc2','svc3','svc4','svc5','svc6','svc7'
]

rs = 1002

AdaBoosts = [
    AdaBoostClassifier(n_estimators=2,random_state=rs),
    AdaBoostClassifier(n_estimators=4,random_state=rs),
    AdaBoostClassifier(n_estimators=5,random_state=rs),
    AdaBoostClassifier(n_estimators=6,random_state=rs),
    AdaBoostClassifier(n_estimators=8,random_state=rs),
    AdaBoostClassifier(n_estimators=10,random_state=rs),
    AdaBoostClassifier(n_estimators=15,random_state=rs),
]
AdaBoost_names = [
    'AdaBoost2','AdaBoost4','AdaBoost5','AdaBoost6','AdaBoost8','AdaBoost10','AdaBoost15'
]


param_grid = {
    "Linear SVM" : {
        'C': [1,5,10,100,500,1000],
    },
    "RBF SVM" : {
        'C': [1,5,10,100,500,1000],
        'gamma': [1e-5, 1e-4,1e-3,1e-2,1e-1],
        'kernel': ['poly', 'sigmoid'],
        'degree': [3,5,8,10]
    },
    "KNN" : {
        'weights': ['distance'],
        'n_neighbors': [1,10,50,100]
    }
}

'''
names = ["Decision Trees", "Neural Networks"]


classifiers = [
    DecisionTreeClassifier(),
    MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(15, 15), random_state=1, verbose=True)
    ]'''

evaluation_names = ["Accuracy","F1 Score","F1_Micro","F1_Macro","F1_Weighted","Log_Loss","Precision","Recall","ROC_AUC"]

evaluation_methods = []

def evaluate(y_true,y_pred):
	return [accuracy_score(y_true, y_pred),
	f1_score(y_true, y_pred, average="binary"),
	#f1_score(y_true, y_pred, average='micro'),
	#f1_score(y_true, y_pred, average='macro'),
	#f1_score(y_true, y_pred, average='weighted'),
	#log_loss(y_true,y_pred),
	precision_score(y_true, y_pred, average="binary"),
	recall_score(y_true, y_pred, average="binary"),
	roc_auc_score(y_true, y_pred)]



def load_dataset(trainfilelist, indexlist):
    x = []
    Y = []
    allfeatures = []
    allindex = []
    names = []
    for i,files in enumerate(trainfilelist):
        f = open(files, 'r')
        f2 = open(indexlist[i], "r")
        names = f.readline()
        names = names.strip().split(" ")[:-1]
        for lines in f:
            features = [float(value) for value in lines.split(' ')]
            #print(features)
            allfeatures.append(features)
        for lines in f2:
            indexes = [int(value) for value in lines.split(' ')]
            allindex.append(indexes)
    # from random import shuffle
    # shuffle(allfeatures)
    print(len(allfeatures[0]))
    for feature in allfeatures:
        x.append(feature[:-1])
        #print(feature[-1])
        Y.append(feature[-1])
    return names,x, Y, allindex


def feature_select(X,y):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape) 
    return X_new, y


def plot_data_and_sample(X,y, sampler):
    # Instanciate a PCA object for the sake of easy visualisation
    pca = PCA(n_components=2)

    # Fit and transform x to visualise inside a 2D feature space
    X_vis = pca.fit_transform(X)

    X_resampled, y_resampled = sampler.fit_sample(X, y)

    print(len(X_resampled), len(y_resampled))


    X_res_vis = []
    for X_res in X_resampled:
        X_res_vis.append(pca.transform(X_res))

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
    ax1.set_title('Original set')

    ax2.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
                edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
    for iy, e in enumerate(X_res_vis):
        ax2.scatter(e[y_resampled[iy] == 1, 0], e[y_resampled[iy] == 1, 1],
                    label="Class #1", alpha=0.5, edgecolor=almost_black,
                    facecolor=np.random.rand(3,), linewidth=0.15)
    ax2.set_title('Easy ensemble')

    plt.show()

    return X_resampled, y_resampled



def plot_ROC_curve(roc_curve):
    false_positive_rate, true_positive_rate, _ = roc_curve
    roc_auc = 0
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_PR_curve(pr_curve):
    precision, recall, _ = pr_curve
    plt.plot(recall, precision, lw=2, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show()



# Utility function to report best scores
def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



def cross_validate(X,y):
    for name, clf in zip(names[1:3], classifiers[1:3]):
        
        scores = cross_val_score(clf, X, y, cv=4)
        print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))



def randomGridSearch(X,y):
    for name, clf in zip(names[1:3], classifiers[1:3]):
        # run randomized search
        n_iter_search = 2
        random_search = RandomizedSearchCV(clf, param_distributions=param_grid[name],
                                           n_iter=n_iter_search)

        start = time()
        random_search.fit(X, y)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.cv_results_)


def gridSearch(X,y, working_dir):
    for name, clf in zip(names[0:1], classifiers[0:1]):
        # run grid search
        clf = GridSearchCV(clf, param_grid=param_grid[name],cv=4, scoring="roc_auc" ,n_jobs=24)

        start = time()
        clf.fit(X, y)
        with open(working_dir + "/grid_best_2" + name + '.pkl', 'wb') as f1:
                pickle.dump(clf, f1)
        print("GridSearchCV took %.2f seconds candidates"
              " parameter settings." % ((time() - start)))
        report(clf.cv_results_)


def normalize_topic_values(X, y):
    X[X<1e-4] = 0
    return X,y



def split_data(X,y, index, frac=0.2):
    from collections import Counter
    from sklearn.utils import shuffle
    import random
    c = Counter()

    n = len(X)

    X=np.asarray(X)
    y = np.asarray(y)
    index = np.asarray(index)



    for i in range(n):
        if(y[i] == 1):
            c[index[i][0]] += 1;
    l = list(c.items())
    l = shuffle(l, random_state=101)

    test_debates = []

    test_size = int(frac* sum(y))

    k = 0
    while(test_size > 0):
        test_debates.append(l[k][0])
        test_size -= l[k][1]
        k +=1

    print(test_size, test_debates)

    X_test = []
    y_test = []
    X_train = []
    y_train = []
    index_test = []
    index_train = []


    for i in np.random.permutation(n):
        if(index[i][0] in test_debates):
            X_test.append(X[i])
            y_test.append(y[i])
            index_test.append(index[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
            index_train.append(index[i])

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    index_test = np.asarray(index_test)
    index_train = np.asarray(index_train)

    print(np.shape(X_train))

    p = np.random.permutation(len(X_train))
    test_p = np.random.permutation(len(X_test))

    return X_train[p], X_test[test_p], y_train[p], y_test[test_p], index_train[p], index_test[test_p]



def evaluate(X_test, y_test, index_test, clf, name, b=True):

    y_hat = clf.predict(X_test)
    report = metrics.classification_report(y_test, y_hat)
    #print(str(score))
    f = open(working_dir + "/" + name + '_report.txt', 'w')
    f.write(name+"\n")
    f.write(report)
    print(report)

    # try:
    #     plot_ROC_curve(metrics.roc_curve(y_test, clf.decision_function(X_test), pos_label=1))
    #     plot_PR_curve(metrics.precision_recall_curve(y_test,clf.decision_function(X_test), pos_label=1 )) 
    # except:
    #     pass

    try:
        y_prob = clf.predict_proba(X_test)[:,1]
    except:
        pass

    ks = [10,20,30,40,50,60,70,80,90,100,200,300,500,1000]

    allscores = rank_scorers.all_score(y_test, y_prob, ks)

        
    for i,k in enumerate(ks):
        print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t", file=f)

    #print(allscores)
    if not b:
        return 
    sent_list = [dataset_utils.get_sentence(idx) for idx in index_test]
    ff = open(working_dir + "/" + name + '_scores.txt', 'w')
    for tag, score, sent in zip(y_test, y_prob, sent_list):
        print(tag, score, sent, sep="\t", file=ff)

    buster_prob = dataset_utils.get_buster_score(index_test)

    allscores_buster = rank_scorers.all_score(y_test, buster_prob, ks)

    # for tag, score, sent in zip(y_test, buster_prob, sent_list):
    #     print(tag, score, sent, sep="\t")
    print("ClaimBuster",file=f)
    for i,k in enumerate(ks):
        print(k,round(allscores_buster[i][0],3),round(allscores_buster[i][1],3),round(allscores_buster[i][2],3), sep="\t", file=f)


def ensemble_train(X,y, working_dir):
    ees = sampler.get_sampler('balance_cascade')
    X_res, y_res = ees.fit_sample(X,y)
    name = "svm_bcascade"

    try:
        with open(working_dir + "/" + name  + '.pkl', 'rb') as f1:
            clf = pickle.load(f1)
    except:
        # scores = cross_val_score(clf, X, y, cv=4, scoring="roc_auc")
        # print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))
        clf = []
        for i in range(len(X_res)):
            print(Counter(y_res[i]))
            clfi = SVC(kernel="linear", probability=True)
            #clfi=AdaBoostClassifier()
            clfi.fit(X_res[i], y_res[i])
            clf.append(clfi)
            scores = cross_val_score(clfi, X_res[i], y_res[i], cv=4, scoring="roc_auc")
            print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))
        with open(working_dir + "/" + name + '.pkl', 'wb') as f1:
            pickle.dump(clf, f1)  
    return clf

def ensemble_predict_proba(clf, X):
    y_proba = []
    for clfi in clf:
        y_probai = clfi.predict_proba(X)[:,-1]
        y_proba.append(y_probai)

    y_proba = np.asarray(y_proba)

    y_proba_mean = np.mean(y_proba, axis=0)

    y_hat = np.round(y_proba_mean)

    return y_proba_mean, y_hat





def main(working_dir, args):
    f_names, X,y, index = load_dataset([workingdir+"/features.ff"], [workingdir+"/index.txt"])

    print(len(X), len(y), f_names)
    
    name = "svm_bcascade"
    X = np.asarray(X)
    y = np.asarray(y)

    X, y = normalize_topic_values(X,y)

    
    X_train, X_test, y_train, y_test, index_train, index_test = split_data(X, y, index)
    # gridSearch(X,y, working_dir)

    '''
    # exit(0)
    #sampler= EasyEnsemble()

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=44)
    #exit(0)

    fr = open(working_dir+"/"+name+"report.txt", "w")
    clf = ensemble_train(X_train, y_train, working_dir)
    y_prob, y_hat = ensemble_predict_proba(clf, X_test)

    report = metrics.classification_report(y_test, y_hat)
    #print(str(score))
    print(report)
    print(report, file=fr)

    #evaluate(X_test, y_test, index_test, clf, name) 

    ks = [10,20,30,40,50,60,70,80,90,100,200,300,500,1000]

    allscores = rank_scorers.all_score(y_test, y_prob, ks)

    
    for i,k in enumerate(ks):
        print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t")
        print(k,round(allscores[i][0],3),round(allscores[i][1],3),round(allscores[i][2],3), sep="\t", file=fr)

    #print(allscores)
    
    sent_list = [dataset_utils.get_sentence(idx) for idx in index_test]
    f = open(working_dir+"/"+name+"ranks.txt", "w")
    for tag, score, sent in zip(y_test, y_prob, sent_list):
        print(tag, score, sent, sep="\t", file=f)

    buster_prob = dataset_utils.get_buster_score(index_test)

    allscores_buster = rank_scorers.all_score(y_test, buster_prob, ks)

    # for tag, score, sent in zip(y_test, buster_prob, sent_list):
    #     print(tag, score, sent, sep="\t")
    print("ClaimBuster")
    for i,k in enumerate(ks):
        print(k,round(allscores_buster[i][0],3),round(allscores_buster[i][1],3),round(allscores_buster[i][2],3), sep="\t")
        print(k,round(allscores_buster[i][0],3),round(allscores_buster[i][1],3),round(allscores_buster[i][2],3), sep="\t", file=fr)


    exit(0)
    '''

    rsampler = RandomUnderSampler(random_state=487)
    X_train, y_train = rsampler.fit_sample(X_train, y_train)
    X_test_s, y_test_s = rsampler.fit_sample(X_test, y_test)
    # pipe_components.append(None)

    # for AdaBoost_name, adaBoost in zip(AdaBoost_names, AdaBoosts):
    #     print(AdaBoost_name)
    #     try:
    #         adaBoost = pickle.load(open(working_dir + "/" + AdaBoost_name+'.pkl','rb'))
    #     except:
    #         adaBoost.fit(X_train,y_train)
    #         pickle.dump(adaBoost,open(working_dir + "/" + AdaBoost_name+'.pkl','wb'))
    #     evaluate(X_test,y_test,index_test,adaBoost,AdaBoost_name)
    #     evaluate(X_test_s,y_test_s,index_test,adaBoost,AdaBoost_name+'sampled',False)



    

    for name, clf in zip(names[10:], classifiers[10:]):
        print(name)
        # pipe_components[-1] = ('classification', clf)
        # clf = im_Pipeline(pipe_components)
        #print(clf)
        try:
            raise Exception('Retrain')
            with open(working_dir + "/" + name + '.pkl', 'rb') as f1:
                clf = pickle.load(f1)
        except:
            # scores = cross_val_score(clf, X, y, cv=4, scoring="roc_auc")
            # print("Name %s ROC_AUC: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))
            clf.fit(X_train, y_train)
            with open(working_dir + "/" + name + '.pkl', 'wb') as f1:
                pickle.dump(clf, f1)  
        evaluate(X_test, y_test, index_test, clf, name)
        evaluate(X_test_s, y_test_s, index_test, clf, name+'sampled',False)
        



if __name__ == '__main__':
    import os
    import sys


    working_dir = workingdir+"/models_final" #os.argv[-1]
    try:
        os.makedirs(working_dir)
    except:
        pass

    arguments = sys.argv[1:]
    args = defaultdict(None)
    for x in arguments:
        x = x.split("=")
        args[x[0].strip("-")] = x[1]

    main(working_dir, args)
