# Nombre: Narciso Beras
# Matricula: 24-EISN-2-026

import gradio as gr
from modelo import CLASES_ES, EMOCIONES_INFO
from detector import predecir_emocion, actualizar_historial, placeholder_html

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Playfair+Display:wght@700&display=swap');
body, .gradio-container { font-family: 'DM Sans', sans-serif !important; }
.titulo-app {
    text-align: center; padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #f0fdf8 0%, #e8f4fd 100%);
    border-radius: 0 0 24px 24px; margin-bottom: 1rem;
}
.titulo-app h1 { font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 700; color: #1D9E75; margin-bottom: 0.3rem; }
.titulo-app p  { color: #888780; font-size: 0.95rem; }
"""

with gr.Blocks(css=CSS, title="Detector de Emociones — Narciso Beras") as app:

    gr.HTML("""
        <div class='titulo-app'>
            <h1>🎭 Detector de Emociones Faciales</h1>
            <p>Narciso Beras &middot; 24-EISN-2-026 &middot; Inteligencia Artificial</p>
            <p style="font-size:0.8rem;color:#aaa;">EfficientNet-B0 &middot; OpenCV + PyTorch + Gradio</p>
        </div>
    """)

    with gr.Tabs():

        # Tab 1: subir imagen desde archivo o portapapeles
        with gr.Tab("📷 Subir imagen"):
            with gr.Row():
                with gr.Column(scale=1):
                    imagen_input = gr.Image(label="Imagen del rostro", type="numpy", height=320, sources=["upload", "clipboard"])
                    btn_analizar = gr.Button("🔍 Analizar emocion", variant="primary", size="lg")
                with gr.Column(scale=1):
                    resultado_img = gr.HTML(value=placeholder_html())

            btn_analizar.click(fn=predecir_emocion, inputs=imagen_input, outputs=resultado_img)

            gr.Markdown("### Las 7 emociones que el modelo reconoce")
            with gr.Row():
                for key, info in EMOCIONES_INFO.items():
                    gr.HTML(f'''
                    <div style="text-align:center;padding:1rem;background:#f8f9fa;border-radius:12px;border:1px solid #e0e0e0;">
                        <div style="font-size:1.8rem;">{info["emoji"]}</div>
                        <div style="font-weight:600;color:{info["color"]};margin-top:0.3rem;">{CLASES_ES[key]}</div>
                        <div style="font-size:0.75rem;color:#888;margin-top:0.3rem;">{info["descripcion"]}</div>
                    </div>''')

        # Tab 2: captura desde camara web
        with gr.Tab("📹 Camara"):
            gr.Markdown("### Captura una foto con la camara y analiza la emocion")
            with gr.Row():
                with gr.Column(scale=1):
                    camara_input = gr.Image(sources=["webcam"], label="Camara", type="numpy", height=320)
                    btn_camara   = gr.Button("🔍 Analizar foto", variant="primary", size="lg")
                with gr.Column(scale=1):
                    resultado_cam = gr.HTML(value=placeholder_html())

            btn_camara.click(fn=predecir_emocion, inputs=camara_input, outputs=resultado_cam)

        # Tab 3: historial de predicciones de la sesion
        with gr.Tab("📊 Historial"):
            gr.Markdown("### Analisis de esta sesion")
            historial_output = gr.HTML(value='<div style="text-align:center;color:#aaa;padding:2rem;">Sin analisis aun.</div>')
            btn_refrescar    = gr.Button("🔄 Refrescar")
            btn_refrescar.click(fn=actualizar_historial, outputs=historial_output)

        with gr.Tab("ℹ️ Acerca de"):
            gr.HTML("""
            <div style="max-width:700px;margin:0 auto;padding:2rem;">
                <h2 style="color:#1D9E75;font-family:serif;">Detector de Emociones Faciales</h2>
                <p style="color:#555;line-height:1.7;">
                    Detecta rostros con <strong>OpenCV</strong> y clasifica 7 emociones
                    usando <strong>EfficientNet-B0</strong> entrenado con <strong>FER-2013</strong>.
                </p>
                <h3 style="color:#378ADD;">Tecnologias</h3>
                <ul style="color:#555;line-height:2;">
                    <li><strong>OpenCV</strong> — Deteccion de rostros</li>
                    <li><strong>PyTorch + timm</strong> — Modelo de deep learning</li>
                    <li><strong>EfficientNet-B0</strong> — Arquitectura principal</li>
                    <li><strong>FER-2013</strong> — Dataset de entrenamiento</li>
                    <li><strong>Gradio</strong> — Interfaz grafica</li>
                </ul>
                <div style="margin-top:2rem;padding:1rem;background:#f0fdf8;border-radius:12px;border-left:4px solid #1D9E75;">
                    <strong>Autor:</strong> Narciso Beras<br>
                    <strong>Matricula:</strong> 24-EISN-2-026<br>
                    <strong>Asignatura:</strong> Inteligencia Artificial<br>
                    <strong>Ano:</strong> 2026
                </div>
            </div>
            """)

if __name__ == "__main__":
    app.launch(share=True)