# Nombre: Narciso Beras
# Matricula: 24-EISN-2-026
# Descripcion: Interfaz principal de la aplicacion - deteccion de emociones faciales

import gradio as gr

EMOCIONES_DEMO = {
    "Alegria":     {"color": "#1D9E75", "descripcion": "La persona parece estar contenta o satisfecha."},
    "Tristeza":    {"color": "#378ADD", "descripcion": "La persona podria estar triste o decepcionada."},
    "Enojo":       {"color": "#D85A30", "descripcion": "La persona parece molesta o frustrada."},
    "Sorpresa":    {"color": "#EF9F27", "descripcion": "La persona parece sorprendida o asombrada."},
    "Miedo":       {"color": "#7F77DD", "descripcion": "La persona parece asustada o ansiosa."},
    "Asco":        {"color": "#639922", "descripcion": "La persona parece sentir rechazo o desagrado."},
    "Neutralidad": {"color": "#888780", "descripcion": "La persona no muestra una emocion clara."},
}

CSS = """
body { font-family: 'Segoe UI', sans-serif; }

.titulo-principal {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
}
.titulo-principal h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #1D9E75;
    margin-bottom: 0.3rem;
}
.titulo-principal p {
    color: #888780;
    font-size: 0.95rem;
}

.tarjeta-resultado {
    background: #f8f9fa;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #e0e0e0;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.historial-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.7rem 1rem;
    border-radius: 10px;
    background: white;
    border: 1px solid #e0e0e0;
    margin-bottom: 8px;
}
.historial-info { flex: 1; }
.historial-emocion { font-weight: 600; font-size: 0.95rem; }
.historial-tiempo { font-size: 0.8rem; color: #888780; }
.historial-confianza { font-size: 0.85rem; font-weight: 600; }

.placeholder-box {
    text-align: center;
    padding: 3rem 1rem;
    color: #aaa;
    border: 2px dashed #e0e0e0;
    border-radius: 16px;
}
"""

def resultado_placeholder():
    return """
    <div class="tarjeta-resultado">
        <div class="placeholder-box">
            <p>Sube una imagen o activa la camara para detectar emociones</p>
        </div>
    </div>
    """

historial_demo = [
    {"emocion": "Alegria",  "color": "#1D9E75", "confianza": "91.2%", "tiempo": "hace 2 min"},
    {"emocion": "Sorpresa", "color": "#EF9F27", "confianza": "78.5%", "tiempo": "hace 5 min"},
    {"emocion": "Neutral",  "color": "#888780", "confianza": "65.0%", "tiempo": "hace 8 min"},
]

def historial_html():
    items = ""
    for h in historial_demo:
        items += f"""
        <div class="historial-item">
            <div class="historial-info">
                <div class="historial-emocion" style="color:{h['color']};">{h['emocion']}</div>
                <div class="historial-tiempo">{h['tiempo']}</div>
            </div>
            <div class="historial-confianza" style="color:{h['color']};">{h['confianza']}</div>
        </div>
        """
    return f'<div style="padding:0.5rem;">{items}</div>'

with gr.Blocks(css=CSS, title="Detector de Emociones") as app:

    gr.HTML("""
        <div class="titulo-principal">
            <h1>Detector de Emociones Faciales</h1>
            <p>Narciso Beras · 24-EISN-2-026 · Inteligencia Artificial</p>
        </div>
    """)

    with gr.Tabs():

        with gr.Tab("Subir imagen"):
            with gr.Row():
                with gr.Column(scale=1):
                    imagen_input = gr.Image(label="Imagen", type="numpy", height=300)
                with gr.Column(scale=1):
                    resultado_img = gr.HTML(value=resultado_placeholder())

            gr.Markdown("### Las 7 emociones que el sistema reconoce")
            with gr.Row():
                for nombre in list(EMOCIONES_DEMO.keys())[:4]:
                    e = EMOCIONES_DEMO[nombre]
                    gr.HTML(f"""
                        <div style="text-align:center;padding:1rem;background:#f8f9fa;
                             border-radius:12px;border:1px solid #e0e0e0;">
                            <div style="font-weight:600;color:{e['color']};margin-top:0.3rem;">{nombre}</div>
                            <div style="font-size:0.8rem;color:#888;margin-top:0.3rem;">{e['descripcion']}</div>
                        </div>
                    """)
            with gr.Row():
                for nombre in list(EMOCIONES_DEMO.keys())[4:]:
                    e = EMOCIONES_DEMO[nombre]
                    gr.HTML(f"""
                        <div style="text-align:center;padding:1rem;background:#f8f9fa;
                             border-radius:12px;border:1px solid #e0e0e0;">
                            <div style="font-weight:600;color:{e['color']};margin-top:0.3rem;">{nombre}</div>
                            <div style="font-size:0.8rem;color:#888;margin-top:0.3rem;">{e['descripcion']}</div>
                        </div>
                    """)

        with gr.Tab("Camara en vivo"):
            gr.Markdown("### Deteccion en tiempo real")
            with gr.Row():
                with gr.Column(scale=1):
                    camara_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Camara",
                        type="numpy",
                        height=300
                    )
                with gr.Column(scale=1):
                    resultado_cam = gr.HTML(value=resultado_placeholder())

        with gr.Tab("Historial"):
            gr.Markdown("### Analisis recientes")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML(value=historial_html())
                with gr.Column(scale=1):
                    gr.HTML("""
                        <div style="padding:1rem;background:#f8f9fa;border-radius:16px;
                             border:1px solid #e0e0e0;">
                            <h4 style="color:#1D9E75;margin-bottom:1rem;">Resumen</h4>
                            <div style="display:flex;justify-content:space-between;
                                 padding:0.5rem 0;border-bottom:1px solid #e0e0e0;">
                                <span style="color:#888;">Total de analisis</span>
                                <strong>3</strong>
                            </div>
                            <div style="display:flex;justify-content:space-between;
                                 padding:0.5rem 0;border-bottom:1px solid #e0e0e0;">
                                <span style="color:#888;">Emocion mas frecuente</span>
                                <strong style="color:#1D9E75;">Alegria</strong>
                            </div>
                            <div style="display:flex;justify-content:space-between;
                                 padding:0.5rem 0;">
                                <span style="color:#888;">Confianza promedio</span>
                                <strong>78.2%</strong>
                            </div>
                        </div>
                    """)

if __name__ == "__main__":
    app.launch()