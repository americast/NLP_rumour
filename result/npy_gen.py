import numpy as np
import json
import pudb
import pandas as pd
import re
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")

f = open("all.json", "r")
all_dict = json.load(f)
f.close()

f = open("../data/Cancer Claim Data - X causes  cancer.csv", "r")
df = pd.read_csv(f)
f.close()

df = df.dropna()
df = df[df.X != 'none']
df = df[df["Medical relevance"] != 'no']

tweets = list(df["Tweet Text"])
Y = list(df["X"])

arr_full = np.zeros((len(tweets), 110, 100))
Y_final = np.zeros((len(tweets)))

i = 0
j = 0
for tweet in tweets:
	arr = np.zeros((110, 100))
	count = 0
	pos = -1
	key_word = Y[i]
	# pu.db
	for each_word in tweet.split():

		# if i==11:
		# 	print(each_word)
		# 	print(each_word.lower())
		# 	print(snowball_stemmer.stem(re.sub('[^A-Za-z0-9]+', '', each_word.lower())))
		# 	print()
		if snowball_stemmer.stem(re.sub('[^A-Za-z0-9]+', '', each_word.lower())) == snowball_stemmer.stem(re.sub('[^A-Za-z0-9]+', '', key_word.split()[0].lower())):
			pos = count
		if snowball_stemmer.stem(re.sub('[^A-Za-z0-9]+', '', each_word.lower())) in all_dict:
			arr[count] = all_dict[snowball_stemmer.stem(re.sub('[^A-Za-z0-9]+', '', each_word.lower()))]
		count+=1
		if (count >= 110):
			break

	if pos != -1:
		arr_full[j,...] = arr
		Y_final[j] = pos
		j+=1
	i+=1

pu.db

np.save("X_0.npy", arr_full[:j,...])
np.save("Y_0.npy", Y_final[:j])

np.save("X_0_val.npy", arr_full[j:,...])
np.save("Y_0_val.npy", Y_final[j:])

# pu.db
