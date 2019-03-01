import numpy as np
import json
import pudb
import re
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
f = open("all.vec", "r")
all_dict = {}

while(1):
	line = f.readline()
	if not line:
		break
	# pu.db
	line = line.split(' ')
	all_dict[snowball_stemmer.stem(re.sub('[^A-Za-z0-9]+', '', line[0].lower()))] = [float(x) for x in line[1:-1]]

f.close()

f = open("all.json", "w")
json.dump(all_dict, f)
f.close()