import requests
from urllib.parse import quote

API_URL = "http://idir-server2.uta.edu:80/factchecker/score_text/"

def buster_api(text):


	print("Len", len(text))
	output = []
	for i in range(0, len(text), 10):
		end = min(len(text), i+10)
		textb = " ".join(text[i:end])
		data = {"text": textb}
		data =  quote(textb)
		response = requests.get(API_URL+data)
		print(response)
		outputb = response.json()
		output.extend(outputb['results'])
	scores = 0
	print("Len Input", lent, "output lent", len(output))
	return output
	# for result in output['results']:
	# 	scores += result['score']
	# 	# print(result['text'])
	# score = scores/len(output['results'])
	# # print(score)

	# return score