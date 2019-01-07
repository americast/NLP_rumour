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


tt = copy(tt_1)
tt.extend(tt_2)
tt.extend(tt_3)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
TT = vectorizer.fit_transform(tt)

TT = TT.todense()

# pu.db

TT_1 = TT[:len_1]
TT = TT[len_1:]

TT_2 = TT[:len_2]
TT = TT[len_2:]

TT_3 = TT

TT_final = TT_1[:int(0.7 * len_1)]
TT_final = np.vstack((TT_final, TT_2[:int(0.7 * len_2)]))
TT_final = np.vstack((TT_final, TT_3[:int(0.7 * len_3)]))

TT_val = TT_1[int(0.7 * len_1):]
TT_val = np.vstack((TT_val, TT_2[int(0.7 * len_2):]))
TT_val = np.vstack((TT_val, TT_3[int(0.7 * len_3):]))

MR_final = mr_1[:int(0.7 * len_1)]
MR_final.extend(mr_2[:int(0.7 * len_2)])
MR_final.extend(mr_3[:int(0.7 * len_3)])

MR_val = mr_1[int(0.7 * len_1):]
MR_val.extend(mr_2[int(0.7 * len_2):])
MR_val.extend(mr_3[int(0.7 * len_3):])

MR_final = np.array(MR_final)
MR_final = (MR_final == "yes").astype('int')

MR_val = np.array(MR_val)
MR_val = (MR_val == "yes").astype('int')

np.save("./data/X_new.npy",TT_final)
np.save("data/Y_new.npy",MR_final)


np.save("./data/X_new_val.npy",TT_val)
np.save("data/Y_new_val.npy",MR_val)

all_ = {"len_1": len_1, "len_2": len_2, "len_3": len_3}
f = open("data/len.json", "w")

json.dump(all_, f)

# pu.db
# makes the passed rows header 
# pd.read_csv("pokemon.csv", header =[1]) 
