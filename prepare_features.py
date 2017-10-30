import os
from extractors import *
from dataset_utils import get_sentence
import numpy

basepath = "/home/bt1/13CS10060/btp"
datapath = basepath+"/ayush_dataset"
workingdir = os.path.join(basepath, "output_all_excluded")


def get_stopwords(filename):
    words = open(basepath + "/ayush_dataset/"+filename, "r")
    return [ word.strip() for word in words.readlines()]

# extractors = [bigrams.bigram_feature,
#  			dependencies.dependenciesVector,
#  			postags.pos_features,
#  			subjective.subjective_feature
#  			]

# extractors_ = [bigrams, dependencies, postags, subjective, topicLDA]


def init():

	## Intiliaze bigrams
	positivefile = os.path.join(datapath, "yesfile.txt")
	
	threshold = 6
	bigrams.load_bigrams(workingdir, threshold, positivefile, os.path.join(datapath, "stopwords.txt"))

	## Initilaize subjective lexicon
	subjective_lexicon = os.path.join(datapath, "subclue.tff")
	subjective.load_lexicon(subjective_lexicon)

	## Initialization complete	


def write_feature_names(output):
	names = []
	for extractor in extractors_:
		ff = extractor.feature_names()
		print(len(ff))
		names.extend(ff)
	names.append("class_label")
	print(len(names))
	print(" ".join(names), file=output)

def generate_features(filename, classid, output):
	
	topicsall = topicLDA.batch_LDATopicVector(filename)
	f = open(filename, "r")
	index = 0
	for line in f:
		features = []
		for extractor in extractors:
			features.extend(extractor(line))
		features.extend(topicsall[index])
		index = index + 1
		if(index % 100 == 0):print(index)
		features.append(classid)
		print(" ".join(map(str, features)), file=output)

	

# init()
# positivefile = os.path.join(datapath, "allyesfile.txt")
# negfile = os.path.join(datapath, "allexcludednofile.txt")
# outputname = os.path.join(workingdir, "features.ff")

# output = open(outputname, "w")
# write_feature_names(output)
# generate_features(positivefile, 1, output)
# generate_features(negfile, 0, output)



def createEmbeddingsFeatureDict():
	print('Starting')
	with open(os.path.join(datapath, "index.txt"),'r') as indexFile:
		index = indexFile.readlines()
	index = [[int(x) for x in line.strip().split(' ')] for line in index]
	with open(os.path.join(datapath, "embeddings.txt"),'w') as f:
		i = 1
		l = len(index)
		for ind in index:
			sent = get_sentence(ind)
			feature = embeddings.features(sent)
			if type(feature) == numpy.float64:
				print(' '.join(['0']*300),file=f)
				i+1
				continue
			feature = " ".join(map(str,feature))
			print(feature,file=f)
			if (i%100 == 0):
				print(i, 'of', l)
			i+=1

createEmbeddingsFeatureDict()