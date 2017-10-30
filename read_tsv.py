import os
import json
import itertools
# from extractors import *
import numpy as np
import pickle

basepath = "/home/bt1/13CS10037/btp_final_from_server"


# filename = os.path.join(basepath, "ayush_dataset", "annotated_single_all.tsv")
# filename_bustered = os.path.join(basepath, "ayush_dataset", "annotated_single_all_withbuster.tsv")

# filename = os.path.join(basepath, "codes", "liar_dataset", "all.tsv")
filename = os.path.join(basepath, "codes", "liar_dataset", "Satyesh", "train_data.tsv")


done = 0
left = 0

# yesfile = open(os.path.join(basepath, "ayush_dataset", "allyesfileindex.txt"), "w")
# nofile = open(os.path.join(basepath, "ayush_dataset", "allnofileindex.txt"), "w")
# exnofile = open(os.path.join(basepath, "ayush_dataset", "allexcludednofileindex.txt"), "w")

def return_embeds(sent):
	feature = embeddings.features(sent)
	if type(feature) == np.float64:
		feature = np.zeros(300)
		print(sent)
	#feature = " ".join(map(str,feature))
	return feature.tolist()


def get_instance():
	f = open(filename, "r")

	colnames = f.readline().strip().split("\t")

	print("Columns in dataset:")
	print(colnames)

	yield colnames

	for line in f:
		cols = line.strip().split("\t")
		d = dict(zip(colnames, cols))
		class_ = 0
		if(d['check_worthy'] == "1"):
			class_ = 1
		yield cols, d['Sentence'], class_



# def get_instance():
# 	f = open(filename, "r")

# 	colnames = f.readline().split("\t")[:-2]

# 	print("Columns in dataset:")
# 	print(colnames)

# 	yield colnames

# 	for line in f:
# 		cols = line.split("\t")[:-2]
# 		d = dict(zip(colnames, cols))
# 		class_ = 0
# 		if(d['Marked'] == "Y"):
# 			class_ = 1
# 		yield cols, d['Sentence'], class_

def get_instance_bustered():
	f = open(filename_bustered, "r")

	colnames = f.readline().split("\t")

	print("Columns in dataset:")
	print(colnames)

	yield colnames

	for line in f:
		cols = line.strip().split("\t")
		d = dict(zip(colnames, cols))
		class_ = 0
		if(d['Marked'] == "Y"):
			class_ = 1
		yield cols, d['Sentence'], class_

def openie_use():
	s = 0

	samples = []
	for line in f:
		cols = line.split("\t")[:-2]
		d = dict(zip(colnames, cols))

		output = tokenizer.openie(d['Sentence'])

		


		if(d['Marked'] == "Y"):
			print(d['Sentence'], d['Marked'])

			for sent in output:
				print (sent.keys())
				ies = sent['openie']
				for ie in ies:
					print(ie['subject'], " | ", ie['object']," | ", ie['relation'])


			input("Press key")





	# d['Embeds'] = return_embeds(d['Sentence'])

	# samples.append(d)
	# if(s == 0):
	# 	print(d)
	# 	input("hehe")
	# 	s = 1




	# if(d['Checked'].lower()[0] == 'y'):
	# 	if(d['Marked']==""):
	# 		left += 1
	# 	else:
	# 		if(d['Marked'] == "Y"):
	# 			print(d['DebateId'], d['ID'], d['Id-1'], file=yesfile)
	# 		elif(d['Marked'] == "N"):
	# 			print(d['DebateId'], d['ID'], d['Id-1'], file=nofile)
	# 		else:
	# 			print(d)
	# 		done += 1
	# else:
	# 	print(d['DebateId'], d['ID'], d['Id-1'], file=nofile)
	# 	print(d['DebateId'], d['ID'], d['Id-1'], file=exnofile)


# 	#print(json.dumps(d))
# data = {"samples":samples}


# import json
# dtafilename = os.path.join(basepath, "ayush_dataset", "data.json")
# with open(dtafilename, 'w') as outfile:
#     json.dump(data, outfile)


