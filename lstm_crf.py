import keras
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import numpy as np
import pandas as pd
import pudb

all = np.load("result/all.npy")
Y = np.load("result/Y.npy")

def pred_acc(y_true, y_pred):

    return K.mean(y_pred)


# f = open("data/Cancer Claim Data - X causes  cancer.csv", "r")
# df = pd.read_csv(f)
# f.close()

# pu.db
# Y = list(df("X"))

input = Input(shape=(110, 100))
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(input)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(1)  # CRF layer
out = crf(model)  # output

model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function)
model.summary()
# pu.db

Y = keras.utils.to_categorical(Y, num_classes=110)
Y = Y.reshape((Y.shape[0], Y.shape[1], 1))

all_train = all[:int(0.8*all.shape[0]),...]
Y_train = Y[:int(0.8*all.shape[0]),...]
Y_train_dense = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))
Y_train_dense = np.argmax(Y_train_dense, axis = -1)


all_test = all[int(0.8*all.shape[0]):,...]
Y_test = Y[int(0.8*all.shape[0]):,...]
# pu.db
Y_test_dense = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))
Y_test_dense = np.argmax(Y_test_dense, axis = -1)

for i in xrange(100):
	print i
	model.fit(all_train, Y_train, batch_size=1000, epochs=5, verbose=1)

	Y_pred_train = model.predict(all_train, batch_size=1000)
	Y_pred_test = model.predict(all_test, batch_size=1000)

	Y_pred_train_dense = np.reshape(Y_pred_train, (Y_pred_train.shape[0], Y_pred_train.shape[1]))
	Y_pred_train_dense = np.argmax(Y_pred_train_dense, axis = -1)

	Y_pred_test_dense = np.reshape(Y_pred_test, (Y_pred_test.shape[0], Y_pred_test.shape[1]))
	Y_pred_test_dense = np.argmax(Y_pred_test_dense, axis = -1)

	train_acc = np.sum(Y_pred_train_dense == Y_train_dense) * 100.0 / len(Y_pred_train_dense)
	val_acc = np.sum(Y_pred_test_dense == Y_test_dense) * 100.0 / len(Y_pred_test_dense)

	print ("Train acc: ", train_acc)
	print ("Val acc: ", val_acc)

	model.save("check.h5")


pu.db
