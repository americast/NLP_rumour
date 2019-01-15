import pandas as pd 
import nltk
from nltk.corpus import stopwords
import sys
reload(sys)
sys.setdefaultencoding('utf8')

stop_words = set(stopwords.words('english'))

f = open("data/tweet_unique.csv","r")
lines = ""

while(1):
	line = f.readline()
	if not line:
		break
	line = line[0:line.rfind("[")]
	lines += line+"\n"


for word in stop_words:
	lines = lines.decode('utf-8').strip().replace(word,"")

f.close()

f = open("data/preprocessed_tweets","w")

f.write(lines)

f.close()


