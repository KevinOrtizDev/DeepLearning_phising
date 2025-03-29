import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization # type: ignore

def train_lstm_model(csv_url):
    # Cargar el archivo CSV
    #../datasets/phishing_email.csv
    try:
        df = pd.read_csv(csv_url)
        print("Archivo CSV cargado correctamente y listo para procesar.")
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return None
    # Extraer las columnas de texto y las etiquetas
    if 'text' not in df.columns or 'label' not in df.columns:
        print("El archivo CSV debe contener las columnas 'text' y 'label'.")
        return None
    texts = df['text_combined'].values  # Todos los textos
    labels = df['label'].values  # Todas las etiquetas

    # Crear la capa de TextVectorization
    vectorizer = TextVectorization(
        max_tokens=10000,         # Máximo número de palabras en el vocabulario
        output_mode='int',        # Salida en forma de enteros (índices)
        output_sequence_length=10 # Longitud de las secuencias de salida
    )

    # Adaptar el vectorizador al conjunto de datos de texto
    vectorizer.adapt(texts)

    # Convertir los textos en secuencias de índices
    vectorized_texts = vectorizer(texts)

    # Ver los primeros 5 textos vectorizados
    print("Textos vectorizados:")
    print(vectorized_texts[:5])
    return vectorized_texts, vectorizer



