# Algoritmos Genéticos (Genetic Algorithms)

Colección de metaheurísticas evolutivas inspiradas en la selección natural para optimización en Machine Learning.

## Algoritmos Disponibles

### 1. AG - Feature Selection
**Archivo:** `AG_feature_selection.py`

Selección de características óptimas usando evolución de cromosomas binarios.

| Gen | Significado |
|----|-------------|
| `1` | Característica incluida |
| `0` | Característica excluida |

**Proceso evolutivo:**
1. Población binaria aleatoria
2. Evaluación via Cross-Validation
3. Selección por torneo
4. Cruce single-point + mutación bit-flip
5. Elitismo

```bash
python AG_feature_selection.py
```

**Resultado típico:**
```
Best Accuracy: ~0.97 con ~12-18 features (de 30 originales)
```

---

### 2. AG - Hyperparameter Optimization
**Archivo:** `AG_hyperparameter_optimization.py`

Búsqueda automática de hiperparámetros MLP (topología, activación, regularización).

| Gen (Parámetro) | Tipo |
|----------------|------|
| `hidden_layer_sizes` | Discreto (arquitecturas predefinidas) |
| `activation` | Categórico (tanh/relu/logistic) |
| `alpha` | Continuo [0.0001, 0.1] |
| `learning_rate_init` | Continuo [0.001, 0.1] |

**Operadores especializados:**
- Cruce uniforme
- Mutación adaptiva por tipo de gen

```bash
python AG_hyperparameter_optimization.py
```

---

### 3. Neuroevolución
**Archivo:** `AG_neuroevolution.py`

Diseño automático de arquitecturas de redes neuronales (NAS - Neural Architecture Search).

```
Cromosoma: [0, 32, 64, 16, 0]
Decodificado: (32, 64, 16) → 3 capas ocultas
```

**Función de aptitud con penalización:**
```
fitness = accuracy - α₁·neuronas - α₂·capas
```

Esto fuerza arquitecturas mínimas eficientes (Occam's razor evolutivo).

```bash
python AG_neuroevolution.py
```

---

## Requisitos

```bash
pip install numpy scikit-learn
```

O con uv:
```bash
uv sync
```

## Operadores Genéticos Comparados

| Operador | Feature Selection | Hyperparameter | Neuroevolución |
|---------|-------------------|-----------------|----------------|
| **Representación** | Binario [0,1] | Híbrido | Entero |
| **Cruce** | Single-point | Uniforme | Single-point |
| **Mutación** | Bit-flip | Adaptiva | Drop/Resize |
| **Selección** | Torneo | Torneo | Torneo |

## Conceptos Fundamentales

### Teorema de No Free Lunch
> No existe un algoritmo que sea óptimo para todos los problemas.

Los AG shines en espacios:
- Discretos/não diferenciables
- Multimodales (múltiples óptimos locales)
- Noirregulares (sin gradientes disponibles)

### Presión Selectiva
```
Baja → Exploración (diversidad)
Alta → Explotación (convergencia rápida)
```

Balance crítico con `TOURNAMENT_SIZE` y `MUTATION_RATE`.

## Pipeline Genérico

```
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Inicializar   → │  Evaluar    │ →  │  Seleccionar│
    │  Población  │    │  Fitness    │    │  Padres     │
    └─────────────┘    └─────────────┘    └─────────────┘
                                                │
       ┌────────────────────────────────────────┘
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Cruce      │ →  │  Mutar      │ →  │  Reemplazar │
│  (crossover)|    │  (mutation) │    │  Poblacion  │
└─────────────┘    └─────────────┘    └─────────────┘
       │
       └────────────────────────────────────────► (repetir por N generaciones)
```

## Evolución vs Enjambre

| Aspecto | Algoritmos Genéticos | PSO / Swarm |
|--------|--------------------|-------------|
| **Memoria** | Poblacional (todo) | Individual + Global |
| **Mezcla** | Cruce (recombinación) | Velocidad (cinética) |
| **Mutación** | Explícita (requerida) | Implícita (ruido) |
| **Convergencia** | Más lenta | Más rápida |


## Referencias

- Holland, J. H. (1975). Adaptation in Natural and Artificial Systems
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning
- Stanley, K. O. (2002). Evolving Neural Networks through Augmenting Topologies