import pandas as pd 
import pudb

a = pd.read_csv(filepath_or_buffer = "./data/Cancer Claim Data - X causes  cancer.csv") 
a = a.drop(['# number', 'X', 'Does X cause cancer '], axis=1)
a = pd.DataFrame.dropna(a)

tt = list(a["Tweet Text"])
mr = list(a["Medical relevance"])

a = pd.read_csv(filepath_or_buffer = "./data/Cancer Claim Data - X prevents cancer.csv") 
a = a.drop(['# number', 'X', 'Does X prevent cancer '], axis=1)
a = pd.DataFrame.dropna(a)

tt.extend(list(a["Tweet Text"]))
mr.extend(list(a["Medical relevance"]))

a = pd.read_csv(filepath_or_buffer = "./data/Cancer Claim Data - X cures cancer .csv") 
a = a.drop(['# number', 'X', 'Does X cure cancer ','Unnamed: 5'], axis=1)
a = pd.DataFrame.dropna(a)

tt.extend(list(a["Tweet Text"]))
mr.extend(list(a["Medical relevance"]))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
TT = vectorizer.fit_transform(tt)

import numpy as np 
mr = np.array(mr)
MR = (mr == "yes").astype('int')

np.save("./data/X.npy",TT.todense()[:int(0.7 * len(MR))])
np.save("data/Y.npy",MR[:int(0.7 * len(MR))])


np.save("./data/X_val.npy",TT.todense()[int(0.7 * len(MR)):])
np.save("data/Y_val.npy",MR[int(0.7 * len(MR)):])
# pu.db
# makes the passed rows header 
# pd.read_csv("pokemon.csv", header =[1]) 
