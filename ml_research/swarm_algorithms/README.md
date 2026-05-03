# Algoritmos de Enjambre (Swarm Intelligence)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013.svg)](https://numpy.org/)
[![PySwarms](https://img.shields.io/badge/PySwarms-1.3+-green.svg)](https://pyswarms.readthedocs.io/)

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

Búsqueda de hiperparámetros óptimos para SVM (C, gamma, epsilon, tol) usando validación cruzada.

```bash
python PSO_hyperparameter_tuning.py
```

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| C | [0.1, 100] | Regularización |
| gamma | [0.0001, 2.0] | Coeficiente kernel RBF |
| epsilon | [0.01, 1.0] | Tolerancia en función de pérdida |
| tol | [1e-5, 1e-1] | Tolerancia de convergencia |

#### 2.2 Optimización de Pesos Neuronales
**Archivo:** `PSO_NN_training_without_backpropagation.py`

Entrenamiento de redes neuronales sin backpropagation - los pesos se optimizan directamente via PSO usando PySwarms. Resuelve el problema CartPole-v1.

```bash
python PSO_NN_training_without_backpropagation.py
```

#### 2.3 Clustering
**Archivo:** `Swarm_clustering.py`

Agrupamiento no supervisado usando PSO para encontrar K centroides.

```bash
python Swarm_clustering.py
```

---

## Guía de Uso Rápida

```bash
# Clone el repositorio y navegue al directorio
cd swarm_algorithms

# Instale las dependencias
pip install -r requirements.txt

# Ejecute cualquier algoritmo directamente
python PSO_hyperparameter_tuning.py      # Optimiza hiperparámetros SVM
python ABC_feature_selection.py         # Selecciona features automáticamente
python PSO_NN_training_without_backpropagation.py  # Entrena red neuronal
python Swarm_clustering.py              # Clustering no supervisado
```

---

## Requisitos

```bash
pip install numpy scikit-learn matplotlib scipy pyswarms gymnasium
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

| Algoritmo | Mejor para | Complejidad |
|----------|-----------|------------|
| ABC | Feature Selection | Media |
| PSO (HPO 4 params) | Hiperparámetros SVM | Media |
| PSO (NN) | CartPole RL | Alta |
| PSO (Clustering) | Unsupervised | Media |


## Referencias

- Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization
- Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization