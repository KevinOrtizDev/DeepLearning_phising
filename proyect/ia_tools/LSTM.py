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

labels = labels.argmax(axis=1)

# Convertir a numpy y luego a tensor
vectorized_texts = vectorized_texts.numpy()

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(vectorized_texts, labels, test_size=0.2, random_state=42)

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Asegurarse de que los datos de entrada tengan la forma correcta para LSTM
X_train = X_train.unsqueeze(-1)  # (batch_size, sequence_length, 1)
X_test = X_test.unsqueeze(-1)    # (batch_size, sequence_length, 1)

# Crear DataLoader para manejo eficiente de batches
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Solo una neurona de salida para clasificación binaria
    
    def forward(self, x):
        # LSTM devuelve dos valores: output y (h_n, c_n)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Usar la última salida de la secuencia
        return out

# Inicializar el modelo
input_size = 1  # Por ejemplo, cada valor en la secuencia tiene una sola característica
hidden_size = 64
output_size = 1  # Para clasificación binaria
model = LSTMModel(input_size, hidden_size, output_size)

y_train = y_train.view(-1, 1)  # Asegurando que las etiquetas estén en la forma correcta (N, 1)
y_test = y_test.view(-1, 1)

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
