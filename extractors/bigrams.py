import os
import json
import pprint 
import numpy as np
import extractors.tokenizer as tokenizer
from collections import Counter
import operator

pp = pprint.PrettyPrinter(indent=4)


stopwords = ["-lrb-","-rrb-", "...","---", "--", "ph"]#get_stopwords("stopwords.txt")
puncts = list("!';\",.?()`$")

stopwords.extend(puncts)
print("Puncts:")
print(stopwords)




def get_stopwords(filename):
    words = open(filename, "r")
    return [ word.strip() for word in words.readlines()]

def count_bigrams(filename):
    source = open(filename, "r")
    cnt = Counter()
    for line in source:
        for sentence in tokenizer.parse(line.strip()):
            tokens = sentence['tokens']
            last = tokens[0]['lemma'].lower()

            for tokeninfo in tokens[1:]:
                if(tokeninfo['lemma'].lower() in stopwords): continue
                bigram = (last, tokeninfo['lemma'].lower())
                cnt[bigram] += 1
                last = tokeninfo['lemma'].lower()
            #cnt[(last, "</s>")] += 1
    return cnt

def generate_bigrams(threshold, filename, stopwords):
    c = count_bigrams(filename)
    stopwords = get_stopwords(stopwords)
    bigrams = {}
    k = 0
    for bigram,cn in c.most_common():
        ## Remove bigrams consisting of only stop words
        if(bigram[0] in stopwords and bigram[1] in stopwords): continue

        ## Break if the threshold is reached
        if(cn < threshold): break
        bigrams[bigram] =  k
        k = k + 1

    return bigrams

def load_bigrams(datapath, threshold, filename, stopwords):
    import pickle
    global bigrams
    try:
        with open(datapath + '/bigrams.pkl', 'rb') as f1:
            bigrams = pickle.load(f1)
    except:
        bigrams = generate_bigrams(threshold, filename, stopwords)
        with open(datapath + '/bigrams.pkl', 'wb') as f1:
            pickle.dump(bigrams, f1)
    print("The bigrams:")
    print(len(bigrams))
    pp.pprint(bigrams)
    return bigrams


def bigram_feature(text):
    parsed = tokenizer.parse(text)
    feature = [0]*len(bigrams)
    for sentence in parsed:
        tokens = sentence['tokens']
        last = tokens[0]['word'].lower()
        for tokeninfo in tokens[1:]:
            if(tokeninfo['word'].lower() in stopwords): continue
            bigram = (last, tokeninfo['lemma'].lower())
            try:
                feature[bigrams[bigram]] += 1
            except:
                pass
            last = tokeninfo['lemma'].lower()

    return feature


def feature_names():
    s_big = sorted(bigrams.items(), key=operator.itemgetter(1))
    return ["bi_"+s[0][0]+"__"+s[0][1] for s in s_big]

def feature_name_type():
    s_big = sorted(bigrams.items(), key=operator.itemgetter(1))
    return [("bi_"+s[0][0]+"__"+s[0][1], 'NUMERIC') for s in s_big]

def features(text):
    return bigram_feature(text)


