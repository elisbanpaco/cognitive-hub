from fastapi import APIRouter, HTTPException
from schemas.data_models import SongFeatures
import joblib
import pandas as pd
import os

router = APIRouter()

# Localización del artefacto generado por el pipeline de MLOps
MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "../../ml_research/predictive_poc/model_artifacts/mood_classifier.pkl"
))

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None
        print(f"Advertencia: Archivo del modelo no encontrado en la ruta {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Error al cargar el modelo predictivo: {e}")

@router.post("/mood")
async def predict_song_mood(features: SongFeatures):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="El modelo predictivo no está disponible en el servidor."
        )
    
    try:
        # Conversión del esquema validado a DataFrame para compatibilidad con scikit-learn
        data_dict = features.model_dump()
        df = pd.DataFrame([data_dict])
        
        # Ejecución de la inferencia
        prediction = model.predict(df)
        
        return {
            "status": "success",
            "predicted_mood": str(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {str(e)}")