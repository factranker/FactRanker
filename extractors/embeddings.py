from gensim.models import KeyedVectors as Word2Vec
import extractors.tokenizer as tokenizer
#import tokenizer
import numpy as np
import os


basepath = "/home/bt1/13CS10037/btp_final_from_server"
datapath = basepath+"/ayush_dataset"

def get_stopwords(filename):
    words = open(filename, "r")
    return [ word.strip() for word in words.readlines()]

model = None
stopwords = get_stopwords(os.path.join(datapath, "stopwords.txt"))

NDIM = 300
def features(text):
	global model
	if(model is None):
		model = Word2Vec.load_word2vec_format('embeddings/google_news_300.bin', binary=True)
		print("Word vector model Loaded")
	parsed = tokenizer.parse(text.strip())
	embeds = []
	for sentence in parsed:
		toks = sentence['tokens']
		for tokeninfo in toks:
			if tokeninfo['word'].lower() in stopwords:
				continue
			try:
				word = tokeninfo['word'].lower()
				embeds.append(model[word].tolist())
			except:
				try:
					embeds.append(model[tokeninfo['lemma'].lower()].tolist())
				except:
					pass

	embeds = np.asarray(embeds)
	if(len(embeds) > 0):
		return np.mean(embeds, axis=0)
	else:
		return np.zeros(NDIM)

def feature_names():
	return ["embed_"+str(i) for i in range(NDIM)]


def feature_name_type():
	return [("embed_"+str(i), "REAL") for i in range(NDIM)]


def sentence_sim(sent1, sent2):
	global model
	if(model is None):
		model = Word2Vec.load_word2vec_format('embeddings/google_news_300.bin', binary=True)
	parsed = tokenizer.parse(sent1.strip())
	sent1_ = [tokeninfo['word'] for sentence in parsed 
		for tokeninfo in sentence['tokens'] if tokeninfo['word'] not in stopwords]
	parsed = tokenizer.parse(sent2.strip())
	sent2_ = [tokeninfo['word'] for sentence in parsed 
		for tokeninfo in sentence['tokens'] if tokeninfo['word'] not in stopwords]

	return  model.wmdistance(sent1_, sent2_)



#print(features("Word is a game"))


