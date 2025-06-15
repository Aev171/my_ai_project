import tensorflow as tf
import pandas as pd
import seaborn as sns
import keras
from keras._tf_keras.keras.layers import Normalization, Dense, InputLayer, Dropout
from keras._tf_keras.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from keras import ops, utils, layers
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.metrics import RootMeanSquaredError
from keras._tf_keras.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\ATA\OneDrive\Masaüstü\python callsma\tubitak\__pycache__\tf\train.csv")

veri = tf.constant(data)
veri = tf.cast(veri, tf.float32)
veri = tf.random.shuffle(veri)

x = veri[:, 3:-1]
y = tf.expand_dims(veri[:, -1], -1)

Train = 0.8
Validation = 0.1
Test = 0.1
Veri_seti = len(x)

x_train = x[:int(Veri_seti * Train)]
y_train = y[:int(Veri_seti * Train)]

x_val = x[int(Veri_seti * Train):int(Veri_seti * (Validation + Train))]
y_val = y[int(Veri_seti * Train):int(Veri_seti * (Validation + Train))]

x_test = x[int(Veri_seti * (Validation + Train)):]
y_test = y[int(Veri_seti * (Validation + Train)):]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

normalizer = Normalization()
normalizer.adapt(x_train)

erken_uyari = EarlyStopping(monitor="val_loss", patience=5)

model = keras.Sequential([
    InputLayer(shape=(8,)),
    normalizer,
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1)
])#modelin katmanları

model.add(Dropout(0.225))  # Daha yüksek bir dropout oranı kullanabilirsin

model.compile(loss=MeanAbsoluteError(),
              optimizer=Adam(learning_rate=0.001),
              metrics=[RootMeanSquaredError()])#modeli yapılandırır

history = model.fit(train_dataset, validation_data=val_dataset, epochs=150, verbose=1, callbacks=[erken_uyari])#modeli eğitir

model.evaluate(val_dataset)

# Tek bir örnek tahmini yapma
print(model.predict(tf.expand_dims(x_test[0], axis=0)))
print(y_test[0])



# Eğitim ve doğrulama kaybını grafiğe dökme
plt.figure(figsize=(12, 6))

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı (Loss)')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.legend()

# RMSE (Root Mean Squared Error) Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['root_mean_squared_error'], label='Eğitim RMSE')
plt.plot(history.history['val_root_mean_squared_error'], label='Doğrulama RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('Eğitim ve Doğrulama RMSE')
plt.legend()

img = plt.imread("model.png")
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")

plt.show()
