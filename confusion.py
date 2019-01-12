import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pudb
import numpy as np

DATA_NUM = input("Enter data no: ")

BATCH_SIZE = 100000


X_val = np.load("data/X_"+str(DATA_NUM)+"_val.npy")
Y_val = np.load("data/Y_"+str(DATA_NUM)+"_val.npy")

# Y_val = keras.utils.to_categorical(Y_val, num_classes = 2)

model = load_model('check_'+str(DATA_NUM)+'.h5')

Y_pred = model.predict(X_val, batch_size=BATCH_SIZE)

Y_pred = np.argmax(Y_pred,axis=-1)


total = len(Y_pred)

_00 = sum((Y_val+Y_pred)==0)

_11 = sum((Y_val+Y_pred)==2)

_01 = sum((Y_val-Y_pred)>0)

_10 = sum((Y_val-Y_pred)<0)

precision = float(_11)/(_11+_01)
recall = float(_11)/(_11+_10)

pu.db
