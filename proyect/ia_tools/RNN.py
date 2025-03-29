import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import sys
sys.path.append("..")
from helper.TextPreprocessingHelper import TextPreprocessingHelper

# Cargar los datos y preprocesarlos
vectorized_texts, labels = TextPreprocessingHelper.train_lstm_model('../datasets/phishing_email.csv')

# Convertir las etiquetas de formato one-hot a 0 o 1
labels = labels.argmax(axis=1)  # Convertir etiquetas one-hot a 0 o 1

# Convertir a numpy y luego a tensor
vectorized_texts = vectorized_texts.numpy()

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(vectorized_texts, labels, test_size=0.2, random_state=42)

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.long)  # Usamos long para índices de palabras
X_test = torch.tensor(X_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Asegurarte de que las etiquetas estén en la forma correcta (N, 1)
y_train = y_train.view(-1, 1)  # Redimensionar a (N, 1)
y_test = y_test.view(-1, 1)

# Crear DataLoader para manejo eficiente de batches
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Definir el modelo RNN (LSTM)
class RNNNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1):
        super(RNNNet, self).__init__()
        
        # Capa de embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Capa LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # Capa de salida
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Embedding de la entrada (transformar índices de palabras en vectores)
        x = self.embedding(x)
        
        # Pasar la entrada a través de la capa LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Tomamos la última salida del LSTM (correspondiente a la última secuencia)
        out = lstm_out[:, -1, :]
        
        # Pasar la salida a través de la capa completamente conectada
        out = self.fc(out)
        return out

# Parámetros
vocab_size = 10000  # El tamaño de tu vocabulario (ajústalo si es necesario)
embedding_dim = 128  # Tamaño del embedding de las palabras
hidden_size = 64     # Número de unidades en el estado oculto de la LSTM
output_size = 1      # Para clasificación binaria (1 valor por ejemplo)

# Inicializar el modelo
model = RNNNet(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, num_classes=output_size)

# Definir el optimizador y la función de pérdida
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # Para clasificación binaria

# Entrenar el modelo
for epoch in range(10):
    model.train()
    correct = 0  # Inicializar contador de aciertos
    total = 0    # Inicializar contador total de ejemplos
    
    for data, labels in train_loader:
        output = model(data)
        loss = criterion(output.squeeze(), labels.squeeze())  # Asegúrate de que las etiquetas y las salidas tengan las mismas dimensiones
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcular predicciones
        predicted = torch.round(torch.sigmoid(output))  # Convierte logits a 0 o 1
        total += labels.size(0)  # Incrementar el total de ejemplos procesados
        correct += (predicted.squeeze() == labels.squeeze()).sum().item()  # Contar cuántas predicciones fueron correctas
    
    # Calcular precisión
    accuracy = correct / total
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# Evaluar el modelo
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        output = model(data)
        predicted = torch.round(torch.sigmoid(output))  # Para obtener 0 o 1 como salida
        total += labels.size(0)
        correct += (predicted.squeeze() == labels.squeeze()).sum().item()

accuracy = correct / total
print(f"Precisión en los datos de prueba: {accuracy:.4f}")
