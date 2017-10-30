import json
import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report
import error_analysis.similarity_n_gram
import pickle
dataset = None

basepath = "/home/bt1/13CS10060/btp"


def initialize_map():
	filename = os.path.join(basepath, "ayush_dataset", "annotated_single_all.tsv")
	return pd.read_csv(filename, sep="\t", header=0)



def get_sentence(index):
	global dataset
	if(dataset is None):
		dataset = initialize_map()

	did, bid, sid = index

	

	result = dataset[dataset.DebateId == did]
	result = result[result.ID == bid]
	result = result[result.Id_1 == sid]
	return result.iloc[0]['Sentence']
	#result = dataset.query('"DebateId"==@did and "ID"==@bid and "Id-1"==@sid')
	#return result['Sentence']



def gen_full_debates(basedir):
	try:
		os.makedirs(basedir)
	except:
		pass
	global dataset
	if(dataset is None):
		dataset = initialize_map()

	for i in range(1,17):
		sents = []
		debate = dataset[dataset.DebateId == i]
		#print(debate)
		for _,row in debate.iterrows():
			#print(row)
			print(row['ID'], row['Id_1'])
			sents.append(row['Sentence'])
		article = " ".join(sents)
		print(article, file=open(basedir+"/debate"+str(i), "w"))

def detect_mismatch(debateNo):
	
	global dataset
	if(dataset is None):
		dataset = initialize_map()


	filename = os.path.join(basepath, "ayush_dataset", "annotated_single_all_withbuster.tsv")
	fp = open(filename, "w")

	columns = ["Checked","Sentence","Marked","By","Speaker","Party","DebateId","ID","Id_1","BScore"]
	for c in columns:
		print(c, end="\t", file=fp)
	print("",file=fp)

	for i in range(1,17):
		busterfile = open(os.path.join(basepath, "ayush_dataset", "buster_scores","debate"+str(i)), "r")
		s = busterfile.readlines()
		sents = []
		debate = dataset[dataset.DebateId == i]
		#print(debate)
		i = 0
		idx = 0
		for index,row in debate.iterrows():
			#print(row)
			#print(row['ID'], row['Id_1'])
			score = s[i].strip().split("\t")[0]
			row['BScore'] = score 

			for c in columns:
				print(row[c], end="\t", file=fp)
			print("",file=fp)


			if(row['Sentence'] != s[i].strip().split("\t")[-1].strip()):
				
				ss = s[i].strip().split("\t")[-1]
				
				if(ss.find(row['Sentence'], idx) != -1):
					# print("subs hai")
					if(len(ss) == ss.find(row['Sentence'], idx) + len(row['Sentence']) ):
						idx = 0
						i += 1
					else: 
						idx = ss.find(row['Sentence'], idx) + len(row['Sentence'])
					# input()
				else:
					print(debateNo)
					print(row['ID'], row['Id_1'])
					print(s[i].strip().split("\t")[-1])
					print(row['Sentence'])
					# print(idx)
					break
			else:
				idx = 0
				i = i + 1
	# filename = os.path.join(basepath, "ayush_dataset", "annotated_single_all_withbuster.tsv")
	# dataset.to_csv(filename, "\t")



def get_buster_score(indexes):
	global dataset
	import pickle
	if(dataset is None):
		dataset = initialize_map()
	fname = os.path.join(basepath, "ayush_dataset","buster_scores", "debate")
	stored = os.path.join(basepath, "ayush_dataset","buster_scores", "scores.pkl")
	try:
		with open(stored,"rb") as f:
			scoremap = pickle.load(f)
	except:
		ids = [3,4,13,14]
		scoremap = {}
		for id in ids:
			fnamei = fname + str(id)
			busterfile = open(fnamei, "r")
			s = busterfile.readlines()
			debate = dataset[dataset.DebateId == id]
			#print(debate)
			i = 0
			for _,row in debate.iterrows():
				#print(row)
				#print(row['ID'], row['Id_1'])
				idx = (row['DebateId'], row['ID'], row['Id_1'])
				scoremap[idx] = s[i].strip().split("\t")[0].strip()
				i = i + 1
		with open(stored, "wb") as f:
			pickle.dump(scoremap, f)


	return [scoremap[(index[0], index[1], index[2])] for index in indexes]


def evaluate_buster():
	global dataset
	import pickle
	if(dataset is None):
		dataset = initialize_map()
	fname = os.path.join(basepath, "ayush_dataset","buster_scores", "debate")
	ids = [3,4,13,14]
	scoremap = {}
	scores = []
	label = []
	for id in ids:
		fnamei = fname + str(id)
		busterfile = open(fnamei, "r")
		s = busterfile.readlines()
		debate = dataset[dataset.DebateId == id]
		#print(debate)
		i = 0
		for _,row in debate.iterrows():
			#print(row)
			#print(row['ID'], row['Id_1'])
			scores.append(s[i].strip().split("\t")[0].strip())
			if(row['Marked'] == "Y"):
				label.append(1)
			else:
				label.append(0)
			idx = (row['DebateId'], row['ID'], row['Id_1'])
			scoremap[idx] = s[i].strip().split("\t")[0].strip()
			i = i + 1

	o = open(os.path.join(basepath, "ayush_dataset","buster_scores", "score_class.txt"), "w")
	for score, cclass in zip(scores, label):
		print(score, cclass, sep="\t", file=o)


def evaluate_buster_():
	o = open(os.path.join(basepath, "ayush_dataset","buster_scores", "score_class.txt"), "r")
	scores = []
	cclass = []
	for l in o.readlines():
		try:
			score, class_ = l.split("\t")
		except:
			print(l)
		scores.append(float(score))
		cclass.append(int(class_))

	scores = np.asarray(scores)
	cclass = np.asarray(cclass)
	yhat = scores >= 0.5
	print(classification_report(cclass, yhat))


def make_buster_scores_for_all():
	fname = os.path.join(basepath, "ayush_dataset","buster_scores", "debate")
	scores = []
	for i in range(1,17):
		fnamei = fname + str(i)
		busterfile = open(fnamei, "r")
		s = busterfile.readlines()
		for l in s:
			l = l.strip()
			try:
				score, text = l.split("\t")
				scores.append((score, text, i))
			except:
				print(i, l)
			
	print(len(scores))
	savename = os.path.join(basepath, "ayush_dataset","buster_scores", "alldebate.pkl")
	fp = open(savename, "wb")
	pickle.dump(scores, fp)





# evaluate_buster_()

# make_buster_scores_for_all()


detect_mismatch(None)





#gen_full_debates(os.path.join(basepath, "ayush_dataset", "debates"))
#detect_mismatch(os.path.join(basepath, "ayush_dataset", "debates"))

#print(get_sentence((11,2,0)))

