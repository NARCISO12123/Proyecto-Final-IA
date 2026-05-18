# Proyecto-de-Final-IA

## Nombre:
Narciso Beras

## Matrícula:
24-EISN-2-026

## Proyecto

DESCRIPCION GENERAL

La aplicacion permite a los usuarios subir imagenes con expresiones faciales
y el sistema detecta e interpreta automaticamente las emociones presentes.
Las emociones que el sistema reconoce son: alegria, tristeza, enojo, sorpresa,
miedo, asco y neutralidad.

El enfoque principal esta orientado a la accesibilidad emocional, brindando
apoyo a personas que tienen dificultad para identificar o interpretar las
emociones de los demas, como ocurre en condiciones como el autismo, la
alexitimia u otras situaciones que afectan la percepcion emocional. El sistema
actua como un asistente visual que traduce expresiones faciales en informacion
clara y comprensible.


MODULO 1: DETECCION Y CLASIFICACION DE EMOCIONES FACIALES

El primer modulo analiza una imagen para localizar los rostros presentes y
clasificar la emocion predominante de cada uno.

Como funciona con Deep Learning / APIs:

Se utilizara un modelo de vision computacional basado en redes neuronales
convolucionales (CNN), implementado con PyTorch. El modelo puede ser
entrenado desde cero o ajustado mediante fine-tuning sobre un modelo
preentrenado. El proceso es el siguiente:

1. Deteccion facial: se localiza y extrae la region del rostro dentro de la
   imagen usando un detector como MTCNN u OpenCV.

2. Clasificacion emocional: el rostro detectado se envia a un modelo CNN
   o a una API de reconocimiento facial (como Microsoft Azure Face API o
   AWS Rekognition), que devuelve la emocion detectada junto con un nivel
   de confianza porcentual.

3. Visualizacion accesible: el sistema muestra la emocion detectada de
   forma clara, usando etiquetas de texto simples, colores representativos
   y una breve descripcion de lo que esa emocion suele significar en un
   contexto social, facilitando su comprension para el usuario.

Como fuente de datos se utilizaran datasets publicos como FER-2013
(Facial Expression Recognition) o AffectNet, que contienen miles de
imagenes etiquetadas con las emociones basicas. Adicionalmente, se
realizara una recopilacion manual progresiva de imagenes, buscando y
descargando fotografias de expresiones faciales una a una para ampliar
y enriquecer el conjunto de datos utilizado por el sistema.


MODULO 2: INTERFAZ Y ASISTENTE DE COMPRENSION EMOCIONAL

El segundo modulo presenta los resultados al usuario a traves de una
interfaz grafica interactiva, que permite subir
imagenes, visualizar las emociones detectadas y acceder a informacion
de apoyo de forma sencilla e intuitiva.

Funcionalidades:

- Descripcion contextual: por cada emocion detectada, el sistema muestra
  una explicacion sencilla de lo que esa expresion puede indicar en una
  conversacion o situacion cotidiana.

- Historial de analisis: se guardan los resultados anteriores para que
  el usuario pueda revisar patrones y aprender progresivamente a
  reconocer emociones con mayor facilidad.

- Modo de aprendizaje: el sistema puede presentar ejercicios visuales
  donde el usuario intenta identificar la emocion antes de revelar el
  resultado, funcionando como una herramienta educativa y de practica.


VALOR Y APLICACION REAL

- Personas con autismo o alexitimia que encuentran dificultad para
  leer las expresiones emocionales de los demas en su vida diaria.
- Terapeutas y educadores especiales que pueden usar la herramienta
  como apoyo visual en sus sesiones de trabajo.
- Familias y cuidadores que desean ayudar a sus seres queridos a
  desarrollar habilidades de reconocimiento emocional.

Este proyecto nace de la conviccion de que la tecnologia puede ser un
puente hacia una comunicacion mas inclusiva y empatica. La combinacion
del reconocimiento facial inteligente con un enfoque accesible convierte
esta herramienta en una solucion con un impacto humano real, que va mas
alla de lo que ofrecen los detectores de emociones convencionales.

