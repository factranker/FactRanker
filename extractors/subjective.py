from nltk.stem.porter import PorterStemmer
import extractors.tokenizer as tokenizer
import operator

stemmer = PorterStemmer()

def load_lexicon(path):
    fi = open(path)

    global lexicon
    lexicon = {}
    for line in fi:
        pros = {}
        props = line.strip().split(" ")
        pros['type'] = props[0].split("=")[-1]
        pros['word'] = props[2].split("=")[-1]
        pros['pos'] = props[3].split("=")[-1]
        pros['stem'] = props[4].split("=")[-1]
        if(pros['word'] in lexicon):
            lexicon[pros['word']].append(pros)
        else:
            lexicon[pros['word']] = [pros]
    return lexicon



def _pos_is_equal(tag, pos):
    if(pos == "verb"):
        return tag[0:2] == "VB"
    if(pos == "noun"):
        return tag[0:2] == "NN"
    if(pos == "adj"):
        return tag[0:2] == "JJ"
    if(pos == "adverb"):
        return tag[0:2] == "RB"
    if(pos == "anypos"):
        return True
    print(pos)
    return True

def subjective_feature(text):
    parsed = tokenizer.parse(text)
    cnt_strong = 0
    cnt_weak = 0
    length = 0
    for sentence in parsed:
        toks = sentence['tokens']
        for tokeninfo in toks:
            length += 1
            word = tokeninfo['word']
            postag = tokeninfo['pos']
            stemmed = stemmer.stem(word)
            if(word in lexicon):
                valids = lexicon[word]
                for valid in valids:
                    if(_pos_is_equal(postag, valid["pos"])):
                        if(valid['type'] == "strongsubj"):
                            cnt_strong = cnt_strong + 1
                        else:
                            cnt_weak =  cnt_weak + 1
                        break
            if(stemmed in lexicon):
                valids = lexicon[stemmed]
                for valid in valids:
                    if(valid['stem'] == "y" and _pos_is_equal(postag, valid["pos"])):
                        if(valid['type'] == "strongsubj"):
                            cnt_strong = cnt_strong + 1
                        else:
                            cnt_weak =  cnt_weak + 1
                        break
    return [cnt_weak, cnt_strong,length ]


def feature_names():
    return ["sub_weak", "sub_strong", "length"]

def feature_name_type():
    return [("sub_weak", 'NUMERIC'), ("sub_strong", 'NUMERIC'), ("length", 'NUMERIC')]

def features(text):
    return subjective_feature(text)