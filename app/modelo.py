# Nombre: Narciso Beras
# Matricula: 24-EISN-2-026

import os
import torch
import timm
from torch import nn

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PATH = os.path.join(BASE_DIR, "models", "modelo_emociones.pth")
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4 emociones seleccionadas
CLASES = ['angry', 'happy', 'sad', 'surprise']

CLASES_ES = {
    'angry':    'Enojo',
    'happy':    'Alegria',
    'sad':      'Tristeza',
    'surprise': 'Sorpresa',
}

EMOCIONES_INFO = {
    'angry':    {'color': '#D85A30', 'emoji': '😠', 'descripcion': 'La persona parece molesta o frustrada.'},
    'happy':    {'color': '#1D9E75', 'emoji': '😄', 'descripcion': 'La persona parece contenta o satisfecha.'},
    'sad':      {'color': '#378ADD', 'emoji': '😢', 'descripcion': 'La persona podria estar triste o decepcionada.'},
    'surprise': {'color': '#EF9F27', 'emoji': '😲', 'descripcion': 'La persona parece sorprendida o asombrada.'},
}

# Umbral minimo de confianza para mostrar una emocion
UMBRAL_CONFIANZA = 0.45


class EmotionModel(nn.Module):
    def __init__(self, num_clases):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_clases)
        )

    def forward(self, x):
        return self.model(x)


def cargar_modelo():
    modelo = EmotionModel(num_clases=len(CLASES)).to(DISPOSITIVO)
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location=DISPOSITIVO, weights_only=True))
    modelo.eval()
    print(f"Modelo cargado | Dispositivo: {DISPOSITIVO}")
    return modelo