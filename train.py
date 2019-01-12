

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import numpy as np

DATA_NUM = 3

BATCH_SIZE = 100000
MAX_ITER = 1000

X_train = np.load("data/X_"+str(DATA_NUM)+".npy")
Y_train = np.load("data/Y_"+str(DATA_NUM)+".npy")

X_val = np.load("data/X_"+str(DATA_NUM)+"_val.npy")
Y_val = np.load("data/Y_"+str(DATA_NUM)+"_val.npy")

Y_train = keras.utils.to_categorical(Y_train, num_classes = 2)
Y_val = keras.utils.to_categorical(Y_val, num_classes = 2)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(1024, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(monitor='val_loss', filepath="check_"+str(DATA_NUM)+".h5", verbose=True,
save_best_only = True)

model.fit(X_train, Y_train,
          epochs=MAX_ITER, shuffle=True, verbose=True, validation_data=(X_val, Y_val),
          batch_size=BATCH_SIZE, callbacks=[checkpointer])
score = model.evaluate(X_val, Y_val, batch_size=BATCH_SIZE)
print(score)

