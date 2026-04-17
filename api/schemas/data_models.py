from pydantic import BaseModel
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
    population_size: Optional[int] = None
    generations: Optional[int] = None
    mutation_rate: Optional[float] = None
    tournament_size: Optional[int] = None
    max_layers: Optional[int] = None
    max_neurons: Optional[int] = None
    dataset: Optional[str] = "breast_cancer"


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
