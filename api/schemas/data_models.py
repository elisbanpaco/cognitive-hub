from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class AGType(str, Enum):
    FEATURE_SELECTION = "feature_selection"
    HYPERPARAMETER_OPT = "hyperparameter_optimization"
    NEUROEVOLUTION = "neuroevolution"


class DatosAG(BaseModel):
    textoA: str
    textoB: str


class DatosAGConfig(BaseModel):
    population_size: int = Field(default=20, ge=2, description="Tamaño de la población (mínimo 2)")
    generations: int = Field(default=15, ge=1, description="Número de generaciones (mínimo 1)")
    mutation_rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Tasa de mutación (entre 0 y 1)")
    tournament_size: int = Field(default=3, ge=2, description="Tamaño del torneo (mínimo 2)")
    max_layers: int = Field(default=5, ge=1, description="Máximo de capas ocultas")
    max_neurons: int = Field(default=128, ge=1, description="Máximo de neuronas por capa")
    dataset: str = Field(default="breast_cancer", description="Nombre del dataset a utilizar")


class AGProgressMessage(BaseModel):
    type: str
    algorithm: Optional[str] = None
    generation: Optional[int] = None
    best_fitness: Optional[float] = None
    best_accuracy: Optional[float] = None
    best_overall_fitness: Optional[float] = None
    best_overall_accuracy: Optional[float] = None
    features_selected: Optional[int] = None
    architecture: Optional[tuple] = None
    history: Optional[List[Dict[str, Any]]] = None
    best_individual: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class SongFeatures(BaseModel):
    danceability: float = Field(..., ge=0.0, le=1.0, description="Capacidad de baile de la pista")
    energy: float = Field(..., ge=0.0, le=1.0, description="Nivel de energía de la pista")
    acousticness: float = Field(..., ge=0.0, le=1.0, description="Nivel de acústica de la pista")
    tempo: float = Field(..., ge=60.0, le=180.0, description="Tempo en BPM")
