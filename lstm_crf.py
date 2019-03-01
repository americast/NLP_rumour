import keras
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import numpy as np
import pandas as pd
import pudb

all = np.load("result/all.npy")
Y = np.load("result/Y.npy")

# f = open("data/Cancer Claim Data - X causes  cancer.csv", "r")
# df = pd.read_csv(f)
# f.close()

pu.db
# Y = list(df("X"))

input = Input(shape=(110, 100))
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(input)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(1)  # CRF layer
out = crf(model)  # output

model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()
# pu.db

Y = keras.utils.to_categorical(Y, num_classes=110)
Y = Y.reshape((Y.shape[0], Y.shape[1], 1))
model.fit(all, Y, batch_size=1000, epochs=100, validation_split=0.2, verbose=1)