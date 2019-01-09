import pandas as pd 
import pudb
from copy import copy
import json
import numpy as np 

a = pd.read_csv(filepath_or_buffer = "./data/Cancer Claim Data - X causes  cancer.csv") 
a = a.drop(['# number', 'X', 'Does X cause cancer '], axis=1)
a = pd.DataFrame.dropna(a)

tt_1 = list(a["Tweet Text"])
mr_1 = list(a["Medical relevance"])
len_1 = len(list(a["Medical relevance"]))

a = pd.read_csv(filepath_or_buffer = "./data/Cancer Claim Data - X prevents cancer.csv") 
a = a.drop(['# number', 'X', 'Does X prevent cancer '], axis=1)
a = pd.DataFrame.dropna(a)

tt_2 = list(a["Tweet Text"])
mr_2 = list(a["Medical relevance"])
len_2 = len(list(a["Medical relevance"]))

a = pd.read_csv(filepath_or_buffer = "./data/Cancer Claim Data - X cures cancer .csv") 
a = a.drop(['# number', 'X', 'Does X cure cancer ','Unnamed: 5'], axis=1)
a = pd.DataFrame.dropna(a)

tt_3 = list(a["Tweet Text"])
mr_3 = list(a["Medical relevance"])
len_3 = len(list(a["Medical relevance"]))


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
TT_1 = vectorizer.fit_transform(tt_1)
TT_2 = vectorizer.fit_transform(tt_2)
TT_3 = vectorizer.fit_transform(tt_3)

TT_1 = TT_1.todense()
TT_2 = TT_2.todense()
TT_3 = TT_3.todense()

# pu.db

TT_1_train = TT_1[:int(0.7 * len_1)]
TT_2_train = TT_2[:int(0.7 * len_2)]
TT_3_train = TT_3[:int(0.7 * len_3)]

TT_1_val = TT_1[int(0.7 * len_1):]
TT_2_val = TT_2[int(0.7 * len_2):]
TT_3_val = TT_3[int(0.7 * len_3):]



MR_1_train = mr_1[:int(0.7 * len_1)]
MR_2_train = mr_2[:int(0.7 * len_2)]
MR_3_train = mr_3[:int(0.7 * len_3)]


MR_1_val = mr_1[int(0.7 * len_1):]
MR_2_val = mr_2[int(0.7 * len_2):]
MR_3_val = mr_3[int(0.7 * len_3):]

MR_1_train = np.array(MR_1_train)
MR_1_train = (MR_1_train == "yes").astype('int')

MR_2_train = np.array(MR_2_train)
MR_2_train = (MR_2_train == "yes").astype('int')

MR_3_train = np.array(MR_3_train)
MR_3_train = (MR_3_train == "yes").astype('int')

MR_1_val = np.array(MR_1_val)
MR_1_val = (MR_1_val == "yes").astype('int')

MR_2_val = np.array(MR_2_val)
MR_2_val = (MR_2_val == "yes").astype('int')

MR_3_val = np.array(MR_3_val)
MR_3_val = (MR_3_val == "yes").astype('int')

np.save("./data/X_1.npy",TT_1_train)
np.save("data/Y_1.npy",MR_1_train)

np.save("./data/X_2.npy",TT_2_train)
np.save("data/Y_2.npy",MR_2_train)

np.save("./data/X_3.npy",TT_3_train)
np.save("data/Y_3.npy",MR_3_train)

np.save("./data/X_1_val.npy",TT_1_val)
np.save("data/Y_1_val.npy",MR_1_val)

np.save("./data/X_2_val.npy",TT_2_val)
np.save("data/Y_2_val.npy",MR_2_val)

np.save("./data/X_3_val.npy",TT_3_val)
np.save("data/Y_3_val.npy",MR_3_val)



all_ = {"len_1": len_1, "len_2": len_2, "len_3": len_3}
f = open("data/len.json", "w")

json.dump(all_, f)

# pu.db
# makes the passed rows header 
# pd.read_csv("pokemon.csv", header =[1]) 
