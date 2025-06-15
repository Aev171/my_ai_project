import numpy  as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import keras
from keras._tf_keras.keras.layers import Dense,InputLayer
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.metrics import RootMeanSquaredError,MeanSquaredError
from sklearn.metrics import confusion_matrix

import itertools
import matplotlib.pyplot as plt
import os.path



train_sample=[]
train_lable=[]

for i in range(50):
    genc=randint(14,63)
    train_sample.append(genc)
    train_lable.append(1)

    yasli=randint(64,100)
    train_sample.append(yasli)
    train_lable.append(0)

for i in range(1000):
    genc=randint(14,63)
    train_sample.append(genc)
    train_lable.append(0)

    yasli=randint(64,100)
    train_sample.append(yasli)
    train_lable.append(1)


train_lable=np.array((train_lable))
train_sample=np.array((train_sample))

train_lable,train_sample=shuffle(train_lable,train_sample)
scaler=MinMaxScaler(feature_range=(0,1))

ölcekli_train_sample=scaler.fit_transform((train_sample.reshape(-1,1)))


test_sample=[]
test_lable=[]

for i in range(10):
    genc=randint(14,63)
    test_sample.append(genc)
    test_lable.append(1)

    yasli=randint(64,100)
    test_sample.append(yasli)
    test_lable.append(0)

for i in range(200):
    genc=randint(14,63)
    test_sample.append(genc)
    test_lable.append(0)

    yasli=randint(64,100)
    test_sample.append(yasli)
    test_lable.append(1)


test_lable=np.array((test_lable))
test_sample=np.array((test_sample))

testlable,trest_sample=shuffle(test_lable,test_sample)
scaler=MinMaxScaler(feature_range=(0,1))

ölcekli_train_sample=scaler.fit_transform((test_sample.reshape(-1,1)))

model=keras.Sequential(
    [
InputLayer(shape=(1,)),
Dense(units=16,activation="relu"),
Dense(units=32,activation="relu"),
Dense(2,activation="softmax")
    ]
)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(ölcekli_train_sample,train_lable,epochs=30,validation_split=0.1,batch_size=10,verbose=2)

tahmin1=model.predict(ölcekli_train_sample,batch_size=10)
tahmin2=np.argmax(tahmin1,-1)


if os.path.isfile("models/deneme1.h5") is False:
    model.save("models/deneme1.h5")

