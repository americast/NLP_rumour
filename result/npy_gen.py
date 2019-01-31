import numpy as np
import json
import pudb
import pandas as pd
import re

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

arr_full = np.zeros((110, 100, len(tweets)))
Y_final = np.zeros((len(tweets)))

i = 0
for tweet in tweets:
	arr = np.zeros((110, 100))
	count = 0
	pos = 0
	key_word = Y[i]
	# pu.db
	for each_word in tweet.split():
		if re.sub('[^A-Za-z0-9]+', '', each_word.lower()) == re.sub('[^A-Za-z0-9]+', '', key_word.lower()):
			pos = i
		if re.sub('[^A-Za-z0-9]+', '', each_word.lower()) in all_dict:
			arr[count] = all_dict[re.sub('[^A-Za-z0-9]+', '', each_word.lower())]
		count+=1
		if (count >= 110):
			break
	# pu.db
	arr_full[:,:,i] = arr
	Y_final[i] = pos
	i+=1

np.save("all.npy", arr_full)
np.save("Y.npy", Y_final)
