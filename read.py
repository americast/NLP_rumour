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

pu.db

# makes the passed rows header 
# pd.read_csv("pokemon.csv", header =[1]) 
