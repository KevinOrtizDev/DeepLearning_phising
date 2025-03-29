import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..")
from helper.TextPreprocessingHelper import TextPreprocessingHelper
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

physical_devices = tf.config.list_physical_devices('GPU')
print("¿TensorFlow puede usar la GPU?:", tf.test.is_gpu_available())
if physical_devices:
    print(f"Se han detectado las siguientes GPU: {physical_devices}")
else:
    print("No se ha detectado ninguna GPU.")


max_words = 10000  # Número máximo de palabras en el vocabulario
max_len = 100  # Longitud máxima de la secuencia (en palabras)


vectorized_texts, labels = TextPreprocessingHelper.train_lstm_model('../datasets/phishing_email.csv')
# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(vectorized_texts.numpy(), labels, test_size=0.2, random_state=42)

X_train = np.expand_dims(X_train, axis=-1)  # Forma: (muestras, características, 1)
X_test = np.expand_dims(X_test, axis=-1)    # Forma: (muestras, características, 1)

# Crear el modelo LSTM
model = Sequential()

# Capa LSTM
model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], 1)))

# Capa Dropout (para evitar sobreajuste)
model.add(Dropout(0.5))

# Capa densa de salida
model.add(Dense(labels.shape[1], activation='softmax'))  # Número de clases en la salida

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precisión en los datos de prueba: {test_acc}")
