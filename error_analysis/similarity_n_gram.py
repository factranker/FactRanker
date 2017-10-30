from nltk.corpus import stopwords
EN_STOPWORDS = stopwords.words('english')
# print [str(word) for word in EN_STOPWORDS]
import random
from extractors import embeddings

functionwords = ['about', 'across', 'against', 'along', 'around', 'at',
                 'behind', 'beside', 'besides', 'by', 'despite', 'down',
                 'during', 'for', 'from', 'in', 'inside', 'into', 'near', 'of',
                 'off', 'on', 'onto', 'over', 'through', 'to', 'toward',
                 'with', 'within', 'without', 'anything', 'everything',
                 'anyone', 'everyone', 'ones', 'such', 'it', 'itself',
                 'something', 'nothing', 'someone', 'the', 'some', 'this',
                 'that', 'every', 'all', 'both', 'one', 'first', 'other',
                 'next', 'many', 'much', 'more', 'most', 'several', 'no', 'a',
                 'an', 'any', 'each', 'no', 'half', 'twice', 'two', 'second',
                 'another', 'last', 'few', 'little', 'less', 'least', 'own',
                 'and', 'but', 'after', 'when', 'as', 'because', 'if', 'what',
                 'where', 'which', 'how', 'than', 'or', 'so', 'before', 'since',
                 'while', 'although', 'though', 'who', 'whose', 'can', 'may',
                 'will', 'shall', 'could', 'be', 'do', 'have', 'might', 'would',
                 'should', 'must', 'here', 'there', 'now', 'then', 'always',
                 'never', 'sometimes', 'usually', 'often', 'therefore',
                 'however', 'besides', 'moreover', 'though', 'otherwise',
                 'else', 'instead', 'anyway', 'incidentally', 'meanwhile']

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(tuple(S[i-c+1:i+1]))
                elif c == longest:
                    lcs_set.add(tuple(S[i-c+1:i+1]))

    return lcs_set

def find_similarity_score(sentence1, sentence2):
	list1 = sentence1.lower().split(' ')
	list2 = sentence2.lower().split(' ')
	list1 = [word for word in list1 if word not in functionwords and word not in EN_STOPWORDS]
	list2 = [word for word in list2 if word not in functionwords and word not in EN_STOPWORDS]
	# lcs_set = lcs(sentence1,sentence2)
	lcs_set = lcs(list1,list2)
	for member in lcs_set:
		return len(lcs_set)*len(member)*len(member)
	return 0

# print find_similarity_score('Hello Priyank! How are you? Do you have the time, to listen to me once?','Hi Priyank! How do you do?')

def test_on_dataset():
	directory = '/home/bt1/13CS10060/btp/ayush_dataset/'
	noFile = directory + 'allnofile.txt'
	yesFile = directory + 'allyesfile.txt'
	with open(yesFile,'r') as yesFile:
		yesFile = yesFile.readlines()
	with open(noFile,'r') as noFile:
		noFile = noFile.readlines()
	random.shuffle(noFile)
	noFile = noFile[:2000]
	count_total = 0
	count = 0
	for sentence in noFile:
		maxScore = 0
		sentence = sentence.strip()
		for yesSentence in yesFile:
			yesSentence = yesSentence.strip()
			score = embeddings.sentence_sim(sentence, yesSentence) #find_similarity_score(sentence,yesSentence)
			if score>maxScore:
				maxScore = score
				similar_sent = yesSentence
		if maxScore>9:
			count+=1
			print(sentence)
			print(similar_sent)
			print(maxScore)
			print()
		count_total += 1
		if count_total%100 == 0:
			print(count_total)
	print(count, 'out of', count_total)

# test_on_dataset()

def test_on_our_results(noFile):
	directory = '/home/bt1/13CS10060/btp/ayush_dataset/'
	yesFile = directory + 'allyesfile.txt'
	with open(yesFile,'r') as yesFile:
		yesFile = yesFile.readlines()
	count_total = 0
	count = 0
	for sentence in noFile:
		maxScore = 0
		sentence = sentence.strip()
		for yesSentence in yesFile:
			yesSentence = yesSentence.strip()
			score = find_similarity_score(sentence,yesSentence)
			if score>maxScore:
				maxScore = score
				similar_sent = yesSentence
		if maxScore>9:
			count+=1
			print(sentence)
			print(similar_sent)
			print(maxScore)
			print()
		count_total += 1
		if count_total%100 == 0:
			print(count_total)
	print(count, 'out of', count_total)



def test_on_our_results_dist(noFile):
	directory = '/home/bt1/13CS10060/btp/ayush_dataset/'
	yesFile = directory + 'allyesfile.txt'
	with open(yesFile,'r') as yesFile:
		yesFile = yesFile.readlines()
	count_total = 0
	count = 0
	for sentence in noFile:
		mindist = 100
		sentence = sentence.strip()
		for yesSentence in yesFile:
			yesSentence = yesSentence.strip()
			dist = embeddings.sentence_sim(sentence, yesSentence)#find_similarity_score(sentence,yesSentence)
			if dist < mindist:
				mindist = dist
				similar_sent = yesSentence
		if mindist <= 2:
			count+=1
			print(sentence)
			print(similar_sent)
			print(mindist)
			print()
		count_total += 1
		if count_total%100 == 0:
			print(count_total)
	print(count, 'out of', count_total)

def get_our_results():
	with open('/home/bt1/13CS10060/btp/codes/error_analysis/ourResults.txt','r') as ourResults:
		ourResults = ourResults.readlines()
	final = []
	for result in ourResults:
		# print(result.split('\t'))
		try:
			annotation,score,sentence = result.split('\t')
			annotation = int(float(annotation))
			if annotation == 0:
				final.append((score,sentence))
		except:
			print('Exception :', result)
	final = sorted(final,reverse=True)
	final = [x[1] for x in final]
	# print(final[:30])
	#final = final[:25]
	# exit(0)
	test_on_our_results_dist(final)

#get_our_results()