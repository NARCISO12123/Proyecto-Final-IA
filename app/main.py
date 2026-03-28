# Nombre: Narciso Beras
# Matricula: 24-EISN-2-026
# Descripcion: Aplicacion de deteccion de emociones faciales con ResNet18 entrenado en FER-2013
#              Interfaz grafica con Gradio. Modelo cargado desde archivo .pth local.

import os
import torch
import timm
import datetime
from collections import Counter
from torch import nn
from PIL import Image
from torchvision import transforms
import gradio as gr

# Ruta absoluta al directorio raiz del proyecto (un nivel arriba de app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PATH = os.path.join(BASE_DIR, "models", "modelo_emociones.pth")

#  Configuracion 
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clases en el mismo orden que se entrenaron en FER-2013
CLASES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

CLASES_ES = {
    'angry':    'Enojo',
    'disgust':  'Asco',
    'fear':     'Miedo',
    'happy':    'Alegria',
    'neutral':  'Neutralidad',
    'sad':      'Tristeza',
    'surprise': 'Sorpresa',
}

EMOCIONES_INFO = {
    'angry':    {'color': '#D85A30', 'emoji': '😠', 'descripcion': 'La persona parece molesta o frustrada.'},
    'disgust':  {'color': '#639922', 'emoji': '🤢', 'descripcion': 'La persona parece sentir rechazo o desagrado.'},
    'fear':     {'color': '#7F77DD', 'emoji': '😨', 'descripcion': 'La persona parece asustada o ansiosa.'},
    'happy':    {'color': '#1D9E75', 'emoji': '😄', 'descripcion': 'La persona parece contenta o satisfecha.'},
    'neutral':  {'color': '#888780', 'emoji': '😐', 'descripcion': 'La persona no muestra una emocion clara.'},
    'sad':      {'color': '#378ADD', 'emoji': '😢', 'descripcion': 'La persona podria estar triste o decepcionada.'},
    'surprise': {'color': '#EF9F27', 'emoji': '😲', 'descripcion': 'La persona parece sorprendida o asombrada.'},
}

#  Transformacion de imagen 
# Misma normalizacion usada en el entrenamiento (test_transform)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#  Definicion del modelo 
class EmotionModel(nn.Module):
    """
    ResNet18 con la capa fc reemplazada para clasificar 7 emociones.
    Se carga con los pesos entrenados en FER-2013.
    """
    def __init__(self, num_clases):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_clases)

    def forward(self, x):
        return self.model(x)

#  Cargar modelo entrenado 
modelo = EmotionModel(num_clases=len(CLASES)).to(DISPOSITIVO)
modelo.load_state_dict(torch.load(MODELO_PATH, map_location=DISPOSITIVO, weights_only=True))
modelo.eval()
print(f"Modelo cargado correctamente | Dispositivo: {DISPOSITIVO}")

#  Historial de sesion 
historial_sesion = []

#  HTML estado vacio 
def placeholder_html():
    return '''
    <div style="background:#f8f9fa;border-radius:16px;padding:2rem;text-align:center;
         border:1px solid #e0e0e0;min-height:200px;display:flex;align-items:center;justify-content:center;">
        <div style="color:#aaa;border:2px dashed #e0e0e0;border-radius:12px;padding:2rem;">
            <div style="font-size:2.5rem;">📷</div>
            <p style="margin-top:0.5rem;">Sube una imagen o activa la camara para detectar emociones</p>
        </div>
    </div>'''

#  Funcion de prediccion 
def predecir_emocion(imagen_np):
    """
    Recibe array numpy, aplica transformacion, ejecuta el modelo
    y retorna HTML con el resultado y las top 3 predicciones.
    """
    if imagen_np is None:
        return placeholder_html()

    try:
        imagen_pil = Image.fromarray(imagen_np.astype('uint8'))
        tensor = transform(imagen_pil).unsqueeze(0).to(DISPOSITIVO)

        with torch.no_grad():
            salida = modelo(tensor)
            probabilidades = torch.softmax(salida, dim=1)[0]

        top3 = torch.topk(probabilidades, 3)
        top3_idx = top3.indices.tolist()
        top3_vals = top3.values.tolist()

        emocion_key = CLASES[top3_idx[0]]
        emocion_es  = CLASES_ES[emocion_key]
        confianza   = top3_vals[0] * 100
        info        = EMOCIONES_INFO[emocion_key]

        # Guardar en historial
        historial_sesion.append({
            'emocion_key': emocion_key,
            'emocion':     emocion_es,
            'confianza':   confianza,
            'color':       info['color'],
            'emoji':       info['emoji'],
            'tiempo':      datetime.datetime.now().strftime('%H:%M:%S'),
        })

        # Barras top 3
        barras_html = ''
        for idx, val in zip(top3_idx, top3_vals):
            nombre_key = CLASES[idx]
            nombre_es  = CLASES_ES[nombre_key]
            pct        = val * 100
            color      = EMOCIONES_INFO[nombre_key]['color']
            barras_html += f'''
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:3px;">
                    <span>{nombre_es}</span>
                    <span style="color:{color};font-weight:600;">{pct:.1f}%</span>
                </div>
                <div style="background:#e0e0e0;border-radius:8px;height:8px;">
                    <div style="background:{color};width:{pct:.1f}%;height:8px;border-radius:8px;"></div>
                </div>
            </div>'''

        html = f'''
        <div style="background:#f8f9fa;border-radius:16px;padding:1.5rem;text-align:center;border:1px solid #e0e0e0;">
            <div style="font-size:3.5rem;">{info["emoji"]}</div>
            <div style="font-size:1.5rem;font-weight:700;color:{info["color"]};margin:0.5rem 0;">{emocion_es}</div>
            <div style="font-size:0.9rem;color:#666;margin-bottom:0.5rem;">{info["descripcion"]}</div>
            <div style="font-size:2.2rem;font-weight:700;color:{info["color"]};">{confianza:.1f}%</div>
            <div style="font-size:0.75rem;color:#aaa;margin-bottom:1.2rem;">confianza</div>
            <div style="width:100%;text-align:left;">
                <div style="font-size:0.8rem;color:#888;margin-bottom:8px;">Top 3 predicciones</div>
                {barras_html}
            </div>
        </div>'''
        return html

    except Exception as e:
        return f'<div style="color:#D85A30;padding:1rem;border-radius:12px;background:#fff3f0;">Error: {str(e)}</div>'

#  Historial 
def actualizar_historial():
    """Genera el HTML del historial con resumen estadistico de la sesion."""
    if not historial_sesion:
        return '<div style="text-align:center;color:#aaa;padding:2rem;">Sin analisis aun en esta sesion.</div>'

    items = ''
    for h in reversed(historial_sesion[-10:]):
        items += f'''
        <div style="display:flex;align-items:center;gap:12px;padding:0.7rem 1rem;
             border-radius:10px;background:white;border:1px solid #e0e0e0;margin-bottom:8px;">
            <div style="font-size:1.5rem;">{h["emoji"]}</div>
            <div style="flex:1;">
                <div style="font-weight:600;color:{h["color"]};">{h["emocion"]}</div>
                <div style="font-size:0.8rem;color:#888;">{h["tiempo"]}</div>
            </div>
            <div style="font-weight:600;color:{h["color"]};">{h["confianza"]:.1f}%</div>
        </div>'''

    total             = len(historial_sesion)
    mas_frecuente_key = Counter(h['emocion_key'] for h in historial_sesion).most_common(1)[0][0]
    mas_frecuente_es  = CLASES_ES[mas_frecuente_key]
    promedio          = sum(h['confianza'] for h in historial_sesion) / total

    resumen = f'''
    <div style="padding:1rem;background:#f8f9fa;border-radius:16px;border:1px solid #e0e0e0;margin-top:1rem;">
        <h4 style="color:#1D9E75;margin-bottom:1rem;">Resumen de sesion</h4>
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #e0e0e0;">
            <span style="color:#888;">Total de analisis</span><strong>{total}</strong>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #e0e0e0;">
            <span style="color:#888;">Emocion mas frecuente</span>
            <strong style="color:#1D9E75;">{mas_frecuente_es} {EMOCIONES_INFO[mas_frecuente_key]["emoji"]}</strong>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;">
            <span style="color:#888;">Confianza promedio</span><strong>{promedio:.1f}%</strong>
        </div>
    </div>'''

    return f'<div style="padding:0.5rem;">{items}{resumen}</div>'

#  CSS 
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Playfair+Display:wght@700&display=swap');
body, .gradio-container { font-family: 'DM Sans', sans-serif !important; }
.titulo-app {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #f0fdf8 0%, #e8f4fd 100%);
    border-radius: 0 0 24px 24px;
    margin-bottom: 1rem;
}
.titulo-app h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem; font-weight: 700; color: #1D9E75; margin-bottom: 0.3rem;
}
.titulo-app p { color: #888780; font-size: 0.95rem; }
"""

#  Interfaz Gradio 
with gr.Blocks(css=CSS, title="Detector de Emociones — Narciso Beras") as app:

    gr.HTML("""
        <div class='titulo-app'>
            <h1>🎭 Detector de Emociones Faciales</h1>
            <p>Narciso Beras &middot; 24-EISN-2-026 &middot; Inteligencia Artificial</p>
            <p style="font-size:0.8rem;color:#aaa;">ResNet18 fine-tuned en FER-2013 &middot; PyTorch + Gradio</p>
        </div>
    """)

    with gr.Tabs():

        #  Tab 1: Subir imagen 
        with gr.Tab("📷 Subir imagen"):
            with gr.Row():
                with gr.Column(scale=1):
                    imagen_input = gr.Image(
                        label="Imagen del rostro",
                        type="numpy",
                        height=320,
                        sources=["upload", "clipboard"]
                    )
                    btn_analizar = gr.Button("🔍 Analizar emocion", variant="primary", size="lg")
                with gr.Column(scale=1):
                    resultado_img = gr.HTML(value=placeholder_html())

            btn_analizar.click(fn=predecir_emocion, inputs=imagen_input, outputs=resultado_img)
            imagen_input.change(fn=predecir_emocion, inputs=imagen_input, outputs=resultado_img)

            gr.Markdown("### Las 7 emociones que el modelo reconoce")
            with gr.Row():
                for key, info in EMOCIONES_INFO.items():
                    gr.HTML(f'''
                    <div style="text-align:center;padding:1rem;background:#f8f9fa;
                         border-radius:12px;border:1px solid #e0e0e0;">
                        <div style="font-size:1.8rem;">{info["emoji"]}</div>
                        <div style="font-weight:600;color:{info["color"]};margin-top:0.3rem;">{CLASES_ES[key]}</div>
                        <div style="font-size:0.75rem;color:#888;margin-top:0.3rem;">{info["descripcion"]}</div>
                    </div>''')

        #  Tab 2: Camara 
        with gr.Tab("📹 Camara"):
            gr.Markdown("### Captura una foto con la camara y analiza la emocion")
            with gr.Row():
                with gr.Column(scale=1):
                    camara_input = gr.Image(
                        sources=["webcam"],
                        label="Camara",
                        type="numpy",
                        height=320
                    )
                    btn_camara = gr.Button("🔍 Analizar foto", variant="primary", size="lg")
                with gr.Column(scale=1):
                    resultado_cam = gr.HTML(value=placeholder_html())

            btn_camara.click(fn=predecir_emocion, inputs=camara_input, outputs=resultado_cam)
            camara_input.change(fn=predecir_emocion, inputs=camara_input, outputs=resultado_cam)

        #  Tab 3: Historial 
        with gr.Tab("📊 Historial"):
            gr.Markdown("### Analisis realizados en esta sesion")
            historial_output = gr.HTML(
                value='<div style="text-align:center;color:#aaa;padding:2rem;">Sin analisis aun.</div>'
            )
            btn_refrescar = gr.Button("🔄 Refrescar historial")
            btn_refrescar.click(fn=actualizar_historial, outputs=historial_output)

        #  Tab 4: Acerca de 
        with gr.Tab("ℹ️ Acerca de"):
            gr.HTML("""
            <div style="max-width:700px;margin:0 auto;padding:2rem;">
                <h2 style="color:#1D9E75;font-family:serif;">Detector de Emociones Faciales</h2>
                <p style="color:#555;line-height:1.7;">
                    Aplicacion que utiliza un modelo <strong>ResNet18</strong> con fine-tuning sobre
                    el dataset <strong>FER-2013</strong> (35,887 imagenes) para detectar 7 emociones
                    faciales basicas en tiempo real.
                </p>
                <h3 style="color:#378ADD;">Tecnologias utilizadas</h3>
                <ul style="color:#555;line-height:2;">
                    <li><strong>PyTorch + timm</strong> — Entrenamiento y ejecucion del modelo</li>
                    <li><strong>ResNet18</strong> — Arquitectura base con fine-tuning</li>
                    <li><strong>FER-2013</strong> — Dataset de expresiones faciales</li>
                    <li><strong>Gradio</strong> — Interfaz grafica interactiva</li>
                </ul>
                <h3 style="color:#378ADD;">Emociones detectadas</h3>
                <p style="color:#555;">Alegria, Tristeza, Enojo, Sorpresa, Miedo, Asco, Neutralidad</p>
                <div style="margin-top:2rem;padding:1rem;background:#f0fdf8;border-radius:12px;border-left:4px solid #1D9E75;">
                    <strong>Autor:</strong> Narciso Beras<br>
                    <strong>Matricula:</strong> 24-EISN-2-026<br>
                    <strong>Asignatura:</strong> Inteligencia Artificial<br>
                    <strong>Ano:</strong> 2026
                </div>
            </div>
            """)

app.launch(share=True)