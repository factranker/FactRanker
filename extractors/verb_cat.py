import extractors.tokenizer as tokenizer

#import tokenizer
from collections import Counter

names = ["report", "knowledge", "belief", "doubt", "perception"]

classword = {
	"report" : ['say' ,'report' ,'tell' ,'told' ,'observe' ,'state','accord' ,'insist' ,'assert' ,'claim' ,'maintain', 'explain' ,'deny'],
	"knowledge" : ['learn' ,'admit' ,'discover' ,'forget' ,'forgot'],
	"belief" : ['think' ,'thought' ,'predict' ,'suggest', 'guess' ,'believe'], 
	"doubt": ['doubt' ,'wonder' ,'ask' ,'hope'],
	"perception": ['sense' ,'hear' ,'feel']
}


def feature_names():
    return ["verbcat_"+ i for i in names]

def feature_name_type():
    return [("verbcat_"+ i, 'NUMERIC') for i in names]


def features(text):
	parsed = tokenizer.parse(text)
	c = Counter()
	for sentence in parsed:
		tokens = sentence['tokens']
		for tokeninfo in tokens:
			tag = tokeninfo['pos']
			if tag[0:2] != "VB": continue
			lemma = tokeninfo['lemma']
			for cat in classword:
				if(lemma in classword[cat]):
					c[cat] += 1


	feature = [0]*len(names)

	for i,name in enumerate(names):
		feature[i] = c[name]

	return feature

print("Testing: Verb_Cat")
print(features("Twenty-nine million people have no health insurance today in America"))


            
            

