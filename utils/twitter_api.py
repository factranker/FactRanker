#coding: utf-8
import oauth, tweepy, sys, locale, threading 
from time import localtime, strftime, sleep


api = None
def init(): 
    global api
    consumer_key = "jx865RXcGn8FzJxJ36WiiSqrT"
    consumer_secret = "Kk1iQGn38KeQ8ZIiih90ovZxfLVUz0pWe0EgIKLOvUypF6nHgv"
    access_key="..."
    access_secret="..."
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    #auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)


def fcount(userlist):
	if(api is None):
		init()
	if type(userlist) is not list:
		userlist = [userlist]
	res = []
	for username in userlist:
		user = api.get_user(username)
		res.append(user.followers_count)
	return res

print("Testing: Twitter API-follower's count")
print(fcount(['agrwalprabhat', "MeetAnimals"]))  ## Answer at time of wrtiting = 24, 2.97M
