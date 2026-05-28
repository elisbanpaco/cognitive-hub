import joblib
import pandas as pd

# 1. Cargar el modelo entrenado desde el artefacto
model_path = 'model_artifacts/mood_classifier.pkl'
try:
    model = joblib.load(model_path)
    print("✅ Modelo cargado exitosamente en memoria.\n")
except FileNotFoundError:
    print("❌ Error: No se encontró el modelo. Ejecuta train.py primero.")
    exit()

# 2. Simulamos las características de dos canciones distintas
# Canción A: Imagina una canción de electrónica o reggaetón (Mucha energía y baile)
cancion_energetica = {
    'danceability': [0.85],
    'energy': [0.92],
    'acousticness': [0.05],
    'tempo': [128.0]
}

# Canción B: Imagina una balada triste a guitarra (Acústica, lenta, baja energía)
cancion_melancolica = {
    'danceability': [0.30],
    'energy': [0.15],
    'acousticness': [0.88],
    'tempo': [75.0]
}

# 3. Función para predecir
def predecir_mood(datos_cancion, nombre="Canción"):
    # Convertimos el diccionario a un DataFrame (como lo espera el modelo)
    df = pd.DataFrame(datos_cancion)
    
    # Hacemos la predicción
    prediccion = model.predict(df)
    
    print(f"--- Evaluando {nombre} ---")
    print(f"Características: {datos_cancion}")
    print(f"🎵 Predicción del Mood: {prediccion[0]}\n")

# 4. Ejecutar la prueba
predecir_mood(cancion_energetica, "Pista 1 (Tipo Electrónica)")
predecir_mood(cancion_melancolica, "Pista 2 (Tipo Balada)")