from utils import verify_and_merge_arff
exit()



import extractors.tokenizer as tk
import pprint
pp = pprint.PrettyPrinter(indent=4)
# import extractors.core_sentence
# from utils import tf_idf
import os
import extractors.sandbox
basepath = "/home/bt1/13CS10060/btp"
filename = os.path.join(basepath, "codes","dataset", "tfidf.pkl")

# tf_idf.trainTFIDF(filename)
exit()

import sys
PATH_TO_SKIPTHOUGHTS = "/home/bt1/13CS10060/btp/skip-thoughts"
sys.path.insert(0, PATH_TO_SKIPTHOUGHTS)

import skipthoughts


model = skipthoughts.load_model()
X = ["I am going to china.", "He is going to Japan"]
vectors = skipthoughts.encode(model, X)
print(scipy.spatial.distance.cosine(X[0],X[1]))

pp.pprint(vectors)


def ner_test():
	text = ""
	while(text != "exit"):

		text = input()
		parsed = tk.ner(text)


		

		pp.pprint(parsed)


