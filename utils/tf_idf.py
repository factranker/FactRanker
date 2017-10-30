from sklearn.feature_extraction.text import TfidfVectorizer
import read_tsv
import extractors.tokenizer as tk
import pickle
import os
import re


def remove_digits(s):
	return re.sub("\d", "#", s)

def preprocessor(text):
	parsed = tk.parse(text)
	return " ".join([ remove_digits(tokeninfo['lemma']) for sentence in parsed for tokeninfo in sentence['tokens']])


def doc_generator():
	instances = read_tsv.get_instance()
	next(instances)
	for instance_ in instances:
		_, text, _ = instance_
		yield preprocessor(text)







def trainTFIDF(savepath):
	transformer = TfidfVectorizer(ngram_range=(1,1), max_df=0.95, min_df=10, norm='l2')
	print("Starting")
	transformer.fit(doc_generator())
	print("Done")
	print(transformer.stop_words_)
	print(transformer.get_feature_names())
	print(transformer.transform(["New York is a great city"]))
	pickle.dump(transformer, open(savepath, "wb"))



if __name__ == '__main__':
	basepath = "/home/bt1/13CS10060/btp"
	filename = os.path.join(basepath, "codes","dataset", "tfidf.pkl")
	trainTFIDF(filename)


