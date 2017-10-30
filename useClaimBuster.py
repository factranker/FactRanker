import requests
import ast
import html.parser
import json
import cgi
import urllib
import glob
import os
import itertools
import time

header = {'Host': 'idir-server2.uta.edu','User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0','Accept': 'text/plain, */*; q=0.01','Accept-Language': 'en-US,en;q=0.5','Accept-Encoding': 'gzip, deflate','Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8','X-CSRFToken': 'ttqRm89p7sjaraZofSuD4RoX4ALJ11LX','X-Requested-With': 'XMLHttpRequest','Referer': 'http://idir-server2.uta.edu/claimbuster/demo','Content-Length': '11571','Cookie': 'csrftoken=ttqRm89p7sjaraZofSuD4RoX4ALJ11LX','Connection': 'keep-alive'}

def create_transcript(text):
	return urllib.parse.urlencode({'transcript':text})

def c(filename, basedir):
	with open(filename,'r') as f:
		text = f.read()

	r = requests.post('http://idir-server2.uta.edu/claimbuster/getCFSScore',data=create_transcript(text),headers=header)
	print (r.status_code)
	r = ast.literal_eval(r.text)
	r = r['chron_output'].split('<br>')
	ff = open(os.path.join(basedir, filename.split("/")[-1]), "w")
	for x in r:
		x = x.split('</span>')
		x[0] = x[0].split('>')[-1]
		print (x[0], x[1], sep="\t", file=ff)
	ff.close()

def nc():
	with open('/home/priyank/Desktop/BTP/DebatesDataAnalysis/ncfile.txt','r') as f:
		text = f.read()

	r = requests.post('http://idir-server2.uta.edu/claimbuster/getCFSScore',data=create_transcript(text),headers=header)
	print (r.status_code)
	r = ast.literal_eval(r.text)
	r = r['ranked_output'].split('<br>')
	for x in r:
		x = x.split('</span>')
		x[0] = x[0].split('>')[-1]
		print (x[0], x[1])

def claim_analyze(sent_list):
	text = " ".join(sent_list)
	# print (text)
	while True:
		try:
			r = requests.post('http://idir-server2.uta.edu/claimbuster/getCFSScore',data=create_transcript(text),headers=header)
			if r.status_code!=200:
				print (r.status_code)
			else:
				break
		except:
			pass
		time.sleep(2)
	r = ast.literal_eval(r.text)
	r = r['chron_output'].split('<br>')
	scores = []
	for x in r:
		x = x.split('</span>')
		x[0] = x[0].split('>')[-1]
		scores.append(x[0])
	return scores

print(claim_analyze(["Here is a dog"]))

basedir = "/home/bt1/13CS10037/btp_final_from_server"

debateset = glob.glob(os.path.join(basedir, "ayush_dataset","debates", "*"))
targetdir = os.path.join(basedir,"ayush_dataset", "buster_scores_new")

for debate in debateset:
	print(debate)
	c(debate, targetdir)

# def peek(iterable):
#     try:
#         first = next(iterable)
#     except StopIteration:
#         return None
#     return first, itertools.chain([first], iterable)

# dataset_file = os.path.join('liar_dataset', 'all.tsv')
# buster_file = os.path.join('liar_dataset', 'buster_all.tsv')


# with open(buster_file, 'w') as buster_file_:
# 	buster_file_.write('Sentence\tClaimBuster_Score\tClass\n')
# 	with open(dataset_file, 'r') as dataset_file:
# 		dataset_file.readline()
# 		count = 0
# 		for line in dataset_file:
# 			sentence = line.strip().split('\t')[0]
# 			klass = line.strip().split('\t')[1]
# 			sent_list = [sentence]
# 			scores = claim_analyze(sent_list)
# 			score = max(scores)
# 			count += 1
# 			buster_file_.write(sentence+'\t'+score+'\t'+klass+'\n')
# 			if count%50==0:
# 				print('{} lines written'.format(count))