import pprint
import scipy
import numpy as np
import os
import pickle
from utils import tf_idf

import sys
pp = pprint.PrettyPrinter(indent=4)




basepath = "/home/bt1/13CS10037/btp_final_from_server"
tfidf_model_path = os.path.join(basepath, "codes","dataset", "tfidf.pkl")

tfidf = None

def encode_tfidf(X):
	global tfidf
	if(tfidf is None):
		tfidf = pickle.load(open(tfidf_model_path,"rb"))

	X = tf_idf.preprocessor(X)
	vec = tfidf.transform([X]).todense()[0]
	vec = np.squeeze(np.asarray(vec))
	# print(np.shape(vec))
	return vec


def features(X):
	return encode_tfidf(X)

def feature_names():
	global tfidf
	if(tfidf is None):
		tfidf = pickle.load(open(tfidf_model_path,"rb"))
	fnames = tfidf.get_feature_names()
	print("No of tdisf size", len(fnames))
	return ["tfidf_"+f for f in fnames]

def feature_name_type():
	return [(f, "REAL") for f in feature_names()]


print("Testing tfidf")
test_text = "1.5 million jobs created during the worst economic time this country has had since the Great Depression while the rest of the country lost 400,000 jobs."
print(features(test_text))