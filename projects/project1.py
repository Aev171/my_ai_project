import pandas  as pd
import tensorflow as tf
import keras
from keras.api.layers import Normalization,InputLayer,Dense
from keras.api.optimizers import Adam
from keras.api.losses import BinaryCrossentropy
from keras.api.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt


earyly_stop=keras.callbacks.EarlyStopping(monitor="loss",patience=10)
#veri hazırlama
raw_data=pd.read_csv(r"C:\Users\ATA\Downloads\transfusion.csv")

def compute_data(data,train_r,test_r,valid_r):
    
    train=data[:int(len(data)*train_r)]
    validation=data[int(len(data) * train_r):int(len(data) * (valid_r + train_r))]
    test=data[int(len(data)*(train_r+valid_r)):]

    return train,validation,test

def expend_data(x,y):
    dataset=tf.data.Dataset.from_tensor_slices((x,y))
    dataset= dataset.shuffle(buffer_size=8,reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

    return dataset


y_data=raw_data.iloc[:,-1]
y_data=tf.expand_dims(y_data,axis=-1)
x_data=raw_data.drop(raw_data.columns[-1],axis=1)


y_train,y_valid,y_test=compute_data(y_data,0.8,0.1,0.1)

x_train,x_valid,x_test=compute_data(x_data,0.8,0.1,0.1)


train_set=expend_data(x_train,y_train)
test_set=expend_data(x_test,y_test)
valid_set=expend_data(x_valid,y_valid)

normalizer=Normalization()
normalizer.adapt(x_train.to_numpy())

model=keras.Sequential([
    normalizer,
Dense(16),
Dense(128),
Dense(128),
Dense(64),
Dense(1,activation="sigmoid")
])

model.compile(
    optimizer=(Adam(learning_rate=0.00001)),
    loss=BinaryCrossentropy(),
    metrics=["accuracy"]
)

history=model.fit(train_set,validation_data=valid_set,epochs=325,callbacks=earyly_stop)

model.evaluate(valid_set)

print(model.predict(x_test.iloc[0]))
print(y_test[0])

pred = model.predict(x_test.iloc[[0]])
predicted_class = int(pred[0][0] > 0.5)
print("Tahmin edilen sınıf:", predicted_class)
print("Gerçek etiket:", y_test.numpy()[0])
print(pred[0][0] > 0.5)
print(pred)

plt.figure(figsize=(12, 6))

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı (Loss)')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim RMSE')
plt.plot(history.history['val_accuracy'], label='Doğrulama RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Eğitim ve Doğrulama RMSE')
plt.legend()

plt.show()
