from __future__ import print_function
import json
import nltk

basepath = "/home/bt1/13CS10060/btp"


sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def read_dataset():
	filename = basepath + "/ayush_dataset/annotated_manual.json" 
	json_data=open(filename).read()

	data = json.loads(json_data)
	tab= "\t"

	checked = [] 
	notchecked = []
	for d in data:
		if(d['checked'] == "Y"):
			sents = sentence_tokenizer.tokenize(d['sent'].strip())
			'''for sent in sents:
				if(len(sent) < 2): continue
				print(sent)
				ans = input()
				if(ans != "y"):
					notchecked.append(sent)
				else:
					checked.append(sent)
					'''
			checked.extend(sents)
		else:
			sents = sentence_tokenizer.tokenize(d['sent'].strip())
			notchecked.extend(sents)

	cfile = open(basepath+"/ayush_dataset/c1file.txt", "w")
	ncfile = open(basepath+"/ayush_dataset/nc1file.txt", "w")
	for s in checked:
		print(s, file=cfile)
	for s in notchecked:
		print(s, file=ncfile)



read_dataset()
