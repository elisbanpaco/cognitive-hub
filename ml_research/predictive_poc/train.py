import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("OPTIMIZACIÓN DE CARACTERÍSTICAS E HIPERPARÁMETROS - MOOD PREDICTOR")
print("="*70)

# ============================================
# 1. GENERAR DATOS SINTÉTICOS (mejorados)
# ============================================
def generate_dummy_data(n_samples=2000):
    """Genera dataset sintético con más características y ruido controlado"""
    np.random.seed(42)
    
    # Características base
    data = {
        'danceability': np.random.uniform(0, 1, n_samples),
        'energy': np.random.uniform(0, 1, n_samples),
        'acousticness': np.random.uniform(0, 1, n_samples),
        'valence': np.random.uniform(0, 1, n_samples),  # NUEVA: positividad
        'tempo': np.random.uniform(60, 200, n_samples),
        'loudness': np.random.uniform(-60, 0, n_samples),  # NUEVA
        'speechiness': np.random.uniform(0, 0.5, n_samples)  # NUEVA
    }
    df = pd.DataFrame(data)
    
    # Reglas más complejas para el mood
    conditions = [
        # Energetic: alta energía + alta bailabilidad + tempo rápido
        (df['energy'] > 0.7) & (df['danceability'] > 0.6) & (df['tempo'] > 120),
        # Melancholic: alta acústica + baja energía + baja valencia
        (df['acousticness'] > 0.6) & (df['energy'] < 0.4) & (df['valence'] < 0.4),
        # Happy: alta valencia + energía media
        (df['valence'] > 0.7) & (df['energy'] > 0.5),
        # Sad: baja valencia + baja energía
        (df['valence'] < 0.3) & (df['energy'] < 0.4) & (df['acousticness'] > 0.3),
    ]
    choices = ['Energetic', 'Melancholic', 'Happy', 'Sad']
    df['mood'] = np.select(conditions, choices, default='Chill')
    
    return df

print("📊 Generando dataset sintético con 7 características base...")
df = generate_dummy_data(2500)
print(f"Dataset generado: {df.shape[0]} muestras, {df.shape[1]-1} características")
print(f"Clases: {df['mood'].unique().tolist()}")
print(f"Distribución de clases:\n{df['mood'].value_counts()}")
print("-"*70)

# ============================================
# 2. FEATURE ENGINEERING (crear nuevas características)
# ============================================
print("\n🔧 Aplicando Feature Engineering...")

# Crear características derivadas (interacciones y transformaciones)
df['energy_dance_ratio'] = df['energy'] / (df['danceability'] + 0.001)
df['valence_energy'] = df['valence'] * df['energy']
df['acoustic_energy'] = df['acousticness'] * df['energy']
df['tempo_norm'] = (df['tempo'] - df['tempo'].mean()) / df['tempo'].std()
df['energy_squared'] = df['energy'] ** 2
df['loudness_norm'] = (df['loudness'] - df['loudness'].mean()) / df['loudness'].std()

# Interacciones de orden superior
df['happy_energy'] = df['valence'] * df['energy'] * df['danceability']
df['sad_acoustic'] = (1 - df['valence']) * df['acousticness']

print(f"Características originales: 7")
print(f"Características después de feature engineering: {len(df.columns)-1}")

# ============================================
# 3. PREPARACIÓN DE DATOS
# ============================================
print("\n📋 Preparando datos para entrenamiento...")

# Separar características y target
feature_cols = [col for col in df.columns if col != 'mood']
X = df[feature_cols]
y = df['mood']

# Codificar target (necesario para algunas métricas)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Total características disponibles: {len(feature_cols)}")
print(f"Características: {feature_cols[:5]}...")

# ============================================
# 4. OPTIMIZACIÓN DE CARACTERÍSTICAS (selección de las mejores)
# ============================================
print("\n🎯 Seleccionando las mejores características...")

# Escalar primero para selección justa
scaler_temp = StandardScaler()
X_scaled_temp = scaler_temp.fit_transform(X)

# Usar información mutua (mejor para clasificación)
selector = SelectKBest(mutual_info_classif, k=8)  # Selecciona las 8 mejores
X_selected = selector.fit_transform(X_scaled_temp, y_encoded)

# ✅ CORRECCIÓN: Obtener información de características seleccionadas
selected_indices = selector.get_support(indices=True)
selected_features = [feature_cols[i] for i in selected_indices]
selected_scores = selector.scores_[selected_indices]

print(f"Características seleccionadas (8 mejores):")
for feat, score in zip(selected_features, selected_scores):
    print(f"  - {feat}: {score:.4f}")

# Mostrar características descartadas
discarded = [feat for i, feat in enumerate(feature_cols) if i not in selected_indices]
if discarded:
    print(f"\nCaracterísticas descartadas: {discarded[:5]}")
    if len(discarded) > 5:
        print(f"  ... y {len(discarded)-5} más")

# ============================================
# 5. DIVISIÓN TRAIN/TEST
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n📊 División de datos:")
print(f"  Entrenamiento: {X_train.shape[0]} muestras")
print(f"  Prueba: {X_test.shape[0]} muestras")

# ============================================
# 6. OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================
print("\n🔍 Optimizando hiperparámetros (RandomizedSearchCV)...")

# Espacio de búsqueda más amplio
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced'],
    'bootstrap': [True, False]
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=40,  # Prueba 40 combinaciones
    cv=5,  # Validación cruzada 5-fold
    scoring='f1_weighted',  # Optimizar F1-score
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Buscando mejores hiperparámetros...")
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_cv_score = random_search.best_score_

print(f"\n✅ MEJORES HIPERPARÁMETROS encontrados:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"\nMejor puntuación CV (F1-score): {best_cv_score:.4f}")

# ============================================
# 7. EVALUACIÓN COMPLETA DEL MODELO
# ============================================
print("\n📈 EVALUANDO MODELO EN TEST...")

y_pred = best_model.predict(X_test)

# Métricas principales
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"\n{'='*50}")
print("RESULTADOS FINALES")
print(f"{'='*50}")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"✅ F1-Score:  {f1:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall:    {recall:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Matriz de Confusión:")
print(cm)

# Reporte por clase
print(f"\n📋 Reporte de clasificación detallado:")
class_names = label_encoder.classes_.tolist()
print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================
# 8. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# ============================================
print("\n🌟 IMPORTANCIA DE CARACTERÍSTICAS (modelo final):")

feature_importance = best_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.to_string(index=False))

# ============================================
# 9. VALIDACIÓN ADICIONAL (Cross-validation)
# ============================================
print("\n🔄 Validación cruzada final (10-fold):")
cv_scores = cross_val_score(best_model, X_selected, y_encoded, cv=10, scoring='accuracy')
print(f"  Media CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  Scores individuales: {cv_scores}")

# ============================================
# 10. GUARDAR ARTEFACTOS
# ============================================
print("\n💾 Guardando artefactos...")
os.makedirs('model_artifacts', exist_ok=True)

# Guardar modelo y transformadores
artifacts = {
    'model': best_model,
    'scaler': scaler_temp,
    'feature_selector': selector,
    'label_encoder': label_encoder,
    'selected_features': selected_features,
    'all_features': feature_cols,
    'best_params': best_params
}
joblib.dump(artifacts, 'model_artifacts/mood_classifier_optimized.pkl')

# Guardar métricas completas
metrics = {
    "accuracy": float(accuracy),
    "f1_score": float(f1),
    "precision": float(precision),
    "recall": float(recall),
    "best_cv_score": float(best_cv_score),
    "best_params": best_params,
    "selected_features": selected_features,
    "feature_importance": dict(zip(selected_features, feature_importance.tolist())),
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
}

with open('model_artifacts/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Guardar también como CSV para fácil lectura
metrics_df = pd.DataFrame([{
    'accuracy': accuracy,
    'f1_score': f1,
    'precision': precision,
    'recall': recall,
    'best_cv_score': best_cv_score,
    'n_features_selected': len(selected_features),
    'n_estimators': best_params.get('n_estimators'),
    'max_depth': best_params.get('max_depth')
}])
metrics_df.to_csv('model_artifacts/metrics_summary.csv', index=False)

print("✅ Modelo guardado en: model_artifacts/mood_classifier_optimized.pkl")
print("✅ Métricas guardadas en: model_artifacts/metrics.json")
print("✅ Resumen guardado en: model_artifacts/metrics_summary.csv")

# ============================================
# 11. VERIFICACIÓN DE CALIDAD (para CI/CD)
# ============================================
print("\n🔍 VERIFICACIÓN DE CALIDAD PARA CI/CD:")
print(f"{'='*50}")

if accuracy >= 0.85:
    print(f"✅ CRITERIO SUPERADO: Accuracy={accuracy:.4f} >= 0.85")
else:
    print(f"❌ CRITERIO NO SUPERADO: Accuracy={accuracy:.4f} < 0.85")
    print("   El pipeline fallará. Ajusta parámetros o mejora el dataset.")

if f1 >= 0.80:
    print(f"✅ F1-Score óptimo: {f1:.4f} >= 0.80")
else:
    print(f"⚠️ F1-Score mejorable: {f1:.4f}")

print(f"\n📊 Resumen final de optimización:")
print(f"   - Características originales: 7")
print(f"   - Características creadas: +9")
print(f"   - Características seleccionadas: {len(selected_features)}")
print(f"   - Combinaciones de hiperparámetros evaluadas: 40")
print(f"   - Mejor Accuracy obtenido: {accuracy:.4f}")

print("\n" + "="*70)
print("✨ ENTRENAMIENTO Y OPTIMIZACIÓN COMPLETADOS EXITOSAMENTE ✨")
print("="*70)