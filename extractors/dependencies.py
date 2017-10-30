import pickle

import extractors.tokenizer as tokenizer

import operator

# def check_pickle():
#     dictOfCheckedDeps = pickle.load(open('newDictOfCheckedDepsPython2.p','rb'))
#     dictOfUnCheckedDeps = pickle.load(open('newDictOfUnCheckedDepsPython2.p','rb'))
#     d = {}
#     for x in dictOfCheckedDeps:
#         d[x] = {'checked':dictOfCheckedDeps[x] , 'unchecked':0}

#     for x in dictOfUnCheckedDeps:
#         if x not in d.keys():
#             d[x] = {'checked':0 , 'unchecked':dictOfUnCheckedDeps[x]}
#         else:
#             d[x]['unchecked'] = dictOfUnCheckedDeps[x]

#     print(len(d))
#     i = 0
#     for key in d:
#         if abs(d[key]['checked']-d[key]['unchecked'])>0.5:
#             print('"'+key+'" : '+str(i)+',')
#             i+=1

mapping = {
    "advcl:out_of" : 0,
    "conj:versus" : 1,
    "conj" : 2,
    "nmod:against" : 3,
    "nmod:out" : 4,
    "nmod:until" : 5,
    "nmod:out_of" : 6,
    "nmod:as" : 7,
    "advcl:as" : 8,
    "acl:of" : 9,
    "acl:relcl" : 10,
    "nmod:thats" : 11,
    "nmod:for" : 12,
    "nmod:into" : 13,
    "nmod:onto" : 14,
    "nmod:save" : 15,
    "nmod:underneath" : 16,
    "neg" : 17,
    "cc" : 18,
    "nmod:off" : 19,
    "xcomp" : 20,
    "nmod:up" : 21,
    "compound" : 22,
    "nmod:among" : 23,
    "acl:compared" : 24,
    "nmod:except" : 25,
    "nmod:according_to" : 26,
    "mark" : 27,
    "nmod:without" : 28,
    "nmod:with" : 29,
    "cop" : 30,
    "acl:including" : 31,
    "nmod:npmod" : 32,
    "conj:as" : 33,
    "nmod:agent" : 34,
    "nmod:regardless_of" : 35,
    "nmod:as_to" : 36,
    "nmod:on" : 37,
    "dep" : 38,
    "nsubjpass" : 39,
    "aux" : 40,
    "advcl:at" : 41,
    "conj:and" : 42,
    "nmod:unlike" : 43,
    "advcl:like" : 44,
    "nmod:if" : 45,
    "advcl:into" : 46,
    "acl:for" : 47,
    "nmod:like" : 48,
    "advcl:across" : 49,
    "conj:not" : 50,
    "advcl:than" : 51,
    "nmod:toward" : 52,
    "nmod" : 53,
    "nmod:throughout" : 54,
    "nmod:instead_of" : 55,
    "nmod:based" : 56,
    "nmod:such" : 57,
    "advcl:from" : 58,
    "nmod:inside" : 59,
    "nmod:between" : 60,
    "nmod:away_from" : 61,
    "nmod:but" : 62,
    "nmod:during" : 63,
    "mwe" : 64,
    "advcl:with" : 65,
    "nmod:around" : 66,
    "advmod" : 67,
    "nmod:beyond" : 68,
    "acl:to" : 69,
    "nmod:about" : 70,
    "nmod:since" : 71,
    "parataxis" : 72,
    "cc:preconj" : 73,
    "nmod:'s" : 74,
    "conj:yet" : 75,
    "nmod:close_to" : 76,
    "nmod:along" : 77,
    "nmod:across" : 78,
    "nmod:from" : 79,
    "ROOT" : 80,
    "acl:in" : 81,
    "nmod:through" : 82,
    "advcl:based_on" : 83,
    "nmod:including" : 84,
    "nmod:behind" : 85,
    "discourse" : 86,
    "nmod:contrary_to" : 87,
    "nmod:by" : 88,
    "nsubj" : 89,
    "nmod:because_of" : 90,
    "advcl:for" : 91,
    "case" : 92,
    "conj:so" : 93,
    "advcl:over" : 94,
    "punct" : 95,
    "nmod:towards" : 96,
    "nmod:above" : 97,
    "nmod:following" : 98,
    "nmod:despite" : 99,
    "advcl:out" : 100,
    "acl:with" : 101,
    "amod" : 102,
    "nmod:to" : 103,
    "advcl" : 104,
    "nmod:that" : 105,
    "nmod:given" : 106,
    "nmod:in_front_of" : 107,
    "advcl:between" : 108,
    "nmod:than" : 109,
    "nmod:at" : 110,
    "nmod:before" : 111,
    "advcl:about" : 112,
    "nummod" : 113,
    "nmod:amongst" : 114,
    "nmod:tmod" : 115,
    "nmod:on_top_of" : 116,
    "nmod:except_for" : 117,
    "nmod:versus" : 118,
    "csubj" : 119,
    "nmod:outside_of" : 120,
    "advcl:on" : 121,
    "advcl:through" : 122,
    "expl" : 123,
    "nmod:in" : 124,
    "conj:but" : 125,
    "nmod:of" : 126,
    "advcl:in" : 127,
    "nmod:within" : 128,
    "advcl:by" : 129,
    "advcl:followed_by" : 130,
    "nmod:per" : 131,
    "nmod:down" : 132,
    "conj:or" : 133,
    "nmod:poss" : 134,
    "det:predet" : 135,
    "nmod:over" : 136,
    "nmod:after" : 137,
    "advcl:to" : 138,
    "nmod:because" : 139,
    "auxpass" : 140,
    "advcl:such" : 141,
    "det" : 142,
    "advcl:of" : 143,
    "nmod:with_respect_to" : 144,
    "nmod:upon" : 145,
    "ccomp" : 146,
    "acl" : 147,
    "nmod:whether" : 148,
    "compound:prt" : 149,
    "nmod:under" : 150,
    "iobj" : 151,
    "nmod:below" : 152,
    "appos" : 153,
    "conj:v." : 154,
    "conj:negcc" : 155,
    "dobj" : 156
}

small_mapping = {
    "compound" : 0,
    "case" : 1,
    "nummod" : 2,
    "amod" : 3,
    "det" : 4,
    "punct" : 5
}

def dependenciesVector(sentence):
    sentence = sentence.strip()
    text = ( sentence )
    output = tokenizer.dependency_parse(text)
    vector = [0]*len(small_mapping)
    # print(output['sentences'][0])
    dependency_type = 'enhancedDependencies' #'collapsed-dependencies'
    for gloss in output['sentences'][0][dependency_type]:
        try:
            vector[small_mapping[gloss['dep']]] += 1
        except:
            pass
    return vector

vector = dependenciesVector('1.5 million jobs created during the worst economic time this country has had since the Great Depression while the rest of the country lost 400,000 jobs.')
print("Testing: Dependencies: ")
print(vector)


def features(sentence):
    return dependenciesVector(sentence)



def feature_names():
    s_big = sorted(small_mapping.items(), key=operator.itemgetter(1))
    return ["dep_"+s[0] for s in s_big]


def feature_name_type():
    s_big = sorted(small_mapping.items(), key=operator.itemgetter(1))
    return [("dep_"+s[0], 'NUMERIC') for s in s_big]
# check_pickle()