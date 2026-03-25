# Nombre: Narciso Beras
# Matricula: 24-EISN-2-026
# Descripcion: Entrenamiento del modelo de deteccion de emociones con FER-2013

import os
import torch
import timm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

#  Configuracion 

CLASES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASES = 7
EPOCAS = 10
LR = 1e-3
BS = 64
SIZE = 48
MEDIA = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando: {DISPOSITIVO}")

#  Transformaciones 

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(SIZE),
    transforms.Resize([SIZE, SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, STD),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize([SIZE, SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, STD),
])

#  Dataset 

class FERDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imagen = self.data[idx]['image']
        etiqueta = self.data[idx]['label']
        if self.transform:
            imagen = self.transform(imagen)
        return imagen, etiqueta

#  Modelo 

class EmotionModel(nn.Module):
    def __init__(self, num_clases):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_clases)

    def forward(self, x):
        return self.model(x)

#  Metrica 

def accuracy(preds, labels):
    _, pred_clases = torch.max(preds, dim=1)
    return (pred_clases == labels).sum().item() / labels.size(0)

#  Entrenamiento 

def entrenar():
    print("Cargando dataset FER-2013...")
    dataset = load_dataset("trpakov/fer-facial-expression-recognition", "base")
    train_data = dataset['train']
    test_data  = dataset['validation']

    train_dataset = FERDataset(train_data, transform=train_transform)
    test_dataset  = FERDataset(test_data,  transform=test_transform)

    cpus = os.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True,  num_workers=cpus)
    test_loader  = DataLoader(test_dataset,  batch_size=BS*2, shuffle=False, num_workers=cpus)

    modelo = EmotionModel(NUM_CLASES).to(DISPOSITIVO)
    perdida_fn = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LR)

    mejor_accuracy = 0.0

    for epoca in range(EPOCAS):
        # Entrenamiento
        modelo.train()
        perdida_train = 0
        acc_train = 0

        for imagenes, etiquetas in tqdm(train_loader, desc=f"Epoca {epoca+1}/{EPOCAS} - Train"):
            imagenes = imagenes.to(DISPOSITIVO)
            etiquetas = etiquetas.to(DISPOSITIVO)

            optimizador.zero_grad()
            preds = modelo(imagenes)
            perdida = perdida_fn(preds, etiquetas)
            perdida.backward()
            optimizador.step()

            perdida_train += perdida.item()
            acc_train += accuracy(preds, etiquetas)

        perdida_train /= len(train_loader)
        acc_train     /= len(train_loader)

        # Evaluacion
        modelo.eval()
        perdida_test = 0
        acc_test = 0

        for imagenes, etiquetas in tqdm(test_loader, desc=f"Epoca {epoca+1}/{EPOCAS} - Test"):
            imagenes = imagenes.to(DISPOSITIVO)
            etiquetas = etiquetas.to(DISPOSITIVO)
            with torch.no_grad():
                preds = modelo(imagenes)
                perdida = perdida_fn(preds, etiquetas)
                perdida_test += perdida.item()
                acc_test += accuracy(preds, etiquetas)

        perdida_test /= len(test_loader)
        acc_test     /= len(test_loader)

        print(f"Epoca {epoca+1} | Train: perdida={perdida_train:.4f} acc={acc_train:.4f} | Test: perdida={perdida_test:.4f} acc={acc_test:.4f}")

        # Guardar el mejor modelo
        if acc_test > mejor_accuracy:
            mejor_accuracy = acc_test
            torch.save(modelo.state_dict(), "models/modelo_emociones.pth")
            print(f"  Modelo guardado con accuracy={mejor_accuracy:.4f}")

    print("Entrenamiento completo.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    entrenar()