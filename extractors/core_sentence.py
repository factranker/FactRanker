import extractors.tokenizer as tk
import pprint
import scipy
import numpy as np
import os
import pickle
from utils import tf_idf
from sklearn.metrics.pairwise import cosine_similarity

import sys
pp = pprint.PrettyPrinter(indent=4)

#Commented by Priyank for now
'''

PATH_TO_SKIPTHOUGHTS = "/home/bt1/13CS10060/btp/skip-thoughts"
sys.path.insert(0, PATH_TO_SKIPTHOUGHTS)

basepath = "/home/bt1/13CS10037/btp_final_from_server"
tfidf_model_path = os.path.join(basepath, "codes","dataset", "tfidf.pkl")

import skipthoughts

model = None
tfidf = None


def encode_skip(X, use_EOS=False):
	global model
	if(model is None):
		model = skipthoughts.load_model()


	vectors = skipthoughts.encode(model, X, verbose=False)
	return vectors


def encode_tfidf(X):
	global tfidf
	if(tfidf is None):
		tfidf = pickle.load(open(tfidf_model_path,"rb"))

	for i in range(len(X)):
		X[i] = tf_idf.preprocessor(X[i])

	return tfidf.transform(X)


def calculate_avg_similarity_block(X, simtype):

	if(simtype == "tfidf"):
		vs = encode_tfidf(X)
	elif(simtype == "skip"):
		vs = encode_skip(X)
	if(len(X) == 1):
		return [0]
	pair_cosine = np.zeros((len(X), len(X)))
	for i in range(len(X)):
		for j in range(len(X)):
			if(i == j): continue
			pair_cosine[i][j] = cosine_similarity(vs[i], vs[j])
	# print(pair_cosine)
	#pair_cosine = 1 - scipy.spatial.distance.cdist(vs, vs, cosine_similarity)
	summed = np.max(pair_cosine, axis=1)
	res = summed 

	return res


def features(X, simtype="tfidf"):
	return calculate_avg_similarity_block(X, simtype)


def feature_name_type(simtype="tfidf"):
	return [('sent_sim_'+simtype, "REAL")]


X = ['Well, thank you.',
	'And I\'m delighted to be here in New Hampshire for this debate.',	
	'You know, the American president has to both keep our families safe and make the economy grow in a way that helps everyone, not just those at the top.',
	'I have a strategy to combat and defeat ISIS without getting us involved in another ground war, and I have plans to raise incomes and deal with a lot of the problems that keep families up at night.']
print("Testing core sentence")

print(features(X, simtype="tfidf"))

# pp.pprint(v)
'''