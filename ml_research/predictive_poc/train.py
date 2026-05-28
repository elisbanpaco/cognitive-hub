import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
import os

# 1. Generar Dataset Sintético (Simulando características de Spotify)
def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'danceability': np.random.uniform(0, 1, n_samples),
        'energy': np.random.uniform(0, 1, n_samples),
        'acousticness': np.random.uniform(0, 1, n_samples),
        'tempo': np.random.uniform(60, 180, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Reglas lógicas simples para asignar el "Mood"
    conditions = [
        (df['energy'] > 0.7) & (df['danceability'] > 0.6),
        (df['acousticness'] > 0.6) & (df['energy'] < 0.5),
    ]
    choices = ['Energetic', 'Melancholic']
    df['mood'] = np.select(conditions, choices, default='Chill')
    
    return df

print("Generando dataset...")
df = generate_dummy_data(1500)

# 2. Preprocesamiento
X = df.drop('mood', axis=1)
y = df['mood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenamiento con Optimización de Hiperparámetros (GridSearchCV)
print("Entrenando modelo y buscando los mejores hiperparámetros...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

# 4. Evaluación del Modelo
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")

# 5. Guardar Artefactos (Modelo y Métricas)
os.makedirs('model_artifacts', exist_ok=True)

# Guardar el modelo serializado
model_path = 'model_artifacts/mood_classifier.pkl'
joblib.dump(best_model, model_path)
print(f"Modelo guardado en: {model_path}")

# Guardar las métricas en un JSON para el pipeline CI/CD
metrics = {
    "accuracy": accuracy,
    "f1_score": f1,
    "best_params": grid_search.best_params_
}

with open('model_artifacts/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("Métricas guardadas en: model_artifacts/metrics.json")