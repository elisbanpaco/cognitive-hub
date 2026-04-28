# Algoritmos de Enjambre (Swarm Intelligence)

Implementaciones de algoritmos metaheurísticos basados en el comportamiento colectivo de sistemas biológicos naturales para problemas de optimización en Machine Learning.

## Algoritmos Disponibles

### 1. ABC - Artificial Bee Colony
**Archivo:** `ABC_feature_selection.py`

Selección automática de características usando el comportamiento de colonies de abejas.

| Componente | Descripción |
|-----------|-------------|
| Abejas Obreras | Explotan fuentes de alimento existentes |
| Abejas Observadoras | Seleccionan fuentes prometedoras probabilísticamente |
| Abejas Exploradoras | Buscan nuevas fuentes cuando las actuales se agotan |

**Uso típico:**
```bash
python ABC_feature_selection.py
```

**Aplicaciones:** Reducción de dimensionalidad, selección de features relevantes.

---

### 2. PSO - Particle Swarm Optimization

#### 2.1 Optimización de Hiperparámetros
**Archivo:** `PSO_hyperparameter_tuning.py`

Búsqueda de hiperparámetros óptimos para SVM (C y gamma).

```bash
python PSO_hyperparameter_tuning.py
```

#### 2.2 Optimización de Pesos Neuronales
**Archivo:** `PSO_weights_optimization.py`

Entrenamiento de redes neuronales sin backpropagation - los pesos se optimizan directamente via PSO.

```bash
python PSO_weights_optimization.py
```

#### 2.3 Clustering
**Archivo:** `Swarm_clustering.py`

Agrupamiento no supervisado usando PSO para encontrar K centroides.

```bash
python Swarm_clustering.py
```

---

## Requisitos

```bash
pip install numpy scikit-learn matplotlib scipy
```

O si usas uv:
```bash
uv sync
```

## Conceptos Clave

### ¿Por qué algoritmos de enjambre?

| Ventaja | Descripción |
|--------|-------------|
| No requieren gradientes | Funciona donde ML simple no puede |
| Búsqueda global | Evitan óptimos locales |
| Paralelizable | Cada partícula es independiente |
| Flexible | Funciona con funciones no diferenciables |

### Ecuación de Velocidad PSO

```
v(t+1) = w·v(t) + c₁·r₁·(pbest - x) + c₂·r₂·(gbest - x)
```

Donde:
- `w` = factor de inercia (exploración)
- `c₁` = componente cognitivo (memoria individual)
- `c₂` = componente social (memoria global)

## Comparativa Rápida

| Algoritmo | Mejor para | Complexidad |
|----------|-----------|------------|
| ABC | Feature Selection | Media |
| PSO (HPO) | Hiperparámetros | Baja |
| PSO (Weights) | NN Training | Alta |
| PSO (Clustering) | Unsupervised | Media |


## Referencias

- Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization
- Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization