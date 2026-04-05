# Nombre: Narciso Beras
# Matricula: 24-EISN-2-026

import os
import datetime
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modelo import CLASES, CLASES_ES, EMOCIONES_INFO, DISPOSITIVO, cargar_modelo

# Rutas de los modelos DNN de OpenCV
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROTOTXT   = os.path.join(BASE_DIR, "models_cv", "deploy.prototxt")
CAFFEMODEL = os.path.join(BASE_DIR, "models_cv", "res10_300x300_ssd_iter_140000.caffemodel")

# Detector DNN — mas preciso que Haar Cascade
DETECTOR_DNN = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

# Misma normalizacion usada en el entrenamiento
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

modelo           = cargar_modelo()
historial_sesion = []


def recortar_rostro(imagen_np):
    """Detecta el rostro con DNN y retorna (recorte, detectado)."""
    h, w = imagen_np.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(imagen_np, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    DETECTOR_DNN.setInput(blob)
    detecciones = DETECTOR_DNN.forward()

    mejor_conf = 0
    mejor_box  = None

    for i in range(detecciones.shape[2]):
        confianza = detecciones[0, 0, i, 2]
        if confianza > 0.5 and confianza > mejor_conf:
            mejor_conf = confianza
            box = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
            mejor_box = box.astype(int)

    if mejor_box is None:
        return imagen_np, False

    x1, y1, x2, y2 = mejor_box
    margen = int(min(x2 - x1, y2 - y1) * 0.2)
    x1 = max(0, x1 - margen)
    y1 = max(0, y1 - margen)
    x2 = min(w, x2 + margen)
    y2 = min(h, y2 + margen)

    return imagen_np[y1:y2, x1:x2], True


def placeholder_html():
    return '''
    <div style="background:#f8f9fa;border-radius:16px;padding:2rem;text-align:center;
         border:1px solid #e0e0e0;min-height:200px;display:flex;align-items:center;justify-content:center;">
        <div style="color:#aaa;border:2px dashed #e0e0e0;border-radius:12px;padding:2rem;">
            <div style="font-size:2.5rem;">📷</div>
            <p style="margin-top:0.5rem;">Sube una imagen o activa la camara para detectar emociones</p>
        </div>
    </div>'''


def predecir_emocion(imagen_np):
    if imagen_np is None:
        return placeholder_html()

    try:
        # Recortar rostro antes de clasificar
        rostro_np, detectado = recortar_rostro(imagen_np)

        aviso = ''
        if not detectado:
            aviso = '''<div style="background:#fff8e1;border:1px solid #ffe082;border-radius:8px;
                 padding:0.5rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:#795548;">
                ⚠️ No se detecto un rostro claro. Usando imagen completa.</div>'''

        tensor = transform(Image.fromarray(rostro_np.astype('uint8'))).unsqueeze(0).to(DISPOSITIVO)

        with torch.no_grad():
            probabilidades = torch.softmax(modelo(tensor), dim=1)[0]

        top3      = torch.topk(probabilidades, 3)
        top3_idx  = top3.indices.tolist()
        top3_vals = top3.values.tolist()

        emocion_key = CLASES[top3_idx[0]]
        emocion_es  = CLASES_ES[emocion_key]
        confianza   = top3_vals[0] * 100
        info        = EMOCIONES_INFO[emocion_key]

        historial_sesion.append({
            'emocion_key': emocion_key,
            'emocion':     emocion_es,
            'confianza':   confianza,
            'color':       info['color'],
            'emoji':       info['emoji'],
            'tiempo':      datetime.datetime.now().strftime('%H:%M:%S'),
        })

        # Barras de las top 3 predicciones
        barras = ''
        for idx, val in zip(top3_idx, top3_vals):
            key   = CLASES[idx]
            pct   = val * 100
            color = EMOCIONES_INFO[key]['color']
            barras += f'''
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:3px;">
                    <span style="color:#333;">{CLASES_ES[key]}</span>
                    <span style="color:{color};font-weight:600;">{pct:.1f}%</span>
                </div>
                <div style="background:#e0e0e0;border-radius:8px;height:8px;">
                    <div style="background:{color};width:{pct:.1f}%;height:8px;border-radius:8px;"></div>
                </div>
            </div>'''

        return f'''
        <div style="background:white;border-radius:16px;padding:1.5rem;text-align:center;border:1px solid #e0e0e0;">
            {aviso}
            <div style="font-size:3.5rem;">{info["emoji"]}</div>
            <div style="font-size:1.5rem;font-weight:700;color:{info["color"]};margin:0.5rem 0;">{emocion_es}</div>
            <div style="font-size:0.9rem;color:#555;margin-bottom:0.5rem;">{info["descripcion"]}</div>
            <div style="font-size:2.2rem;font-weight:700;color:{info["color"]};">{confianza:.1f}%</div>
            <div style="font-size:0.75rem;color:#666;margin-bottom:1.2rem;">confianza</div>
            <div style="width:100%;text-align:left;">
                <div style="font-size:0.8rem;color:#555;margin-bottom:8px;">Top 3 predicciones</div>
                {barras}
            </div>
            <div style="font-size:0.75rem;color:#666;margin-top:1rem;">
                {'✅ Rostro detectado' if detectado else '⚠️ Sin deteccion de rostro'}
            </div>
        </div>'''

    except Exception as e:
        return f'<div style="color:#D85A30;padding:1rem;border-radius:12px;background:#fff3f0;">Error: {str(e)}</div>'


def actualizar_historial():
    if not historial_sesion:
        return '<div style="text-align:center;color:#aaa;padding:2rem;">Sin analisis aun.</div>'

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
    promedio          = sum(h['confianza'] for h in historial_sesion) / total

    return f'''<div style="padding:0.5rem;">{items}
    <div style="padding:1rem;background:#f8f9fa;border-radius:16px;border:1px solid #e0e0e0;margin-top:1rem;">
        <h4 style="color:#1D9E75;margin-bottom:1rem;">Resumen de sesion</h4>
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #e0e0e0;">
            <span style="color:#888;">Total</span><strong>{total}</strong>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #e0e0e0;">
            <span style="color:#888;">Mas frecuente</span>
            <strong style="color:#1D9E75;">{CLASES_ES[mas_frecuente_key]} {EMOCIONES_INFO[mas_frecuente_key]["emoji"]}</strong>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;">
            <span style="color:#888;">Confianza promedio</span><strong>{promedio:.1f}%</strong>
        </div>
    </div></div>'''