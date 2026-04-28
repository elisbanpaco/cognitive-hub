import numpy as np
import random
import warnings
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# 1. Dataset complejo que requiere una buena arquitectura
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, 
    n_redundant=5, n_classes=2, random_state=42
)

print(f"Dataset Complejo: {X.shape[1]} características, {len(np.unique(y))} clases")
X_scaled = StandardScaler().fit_transform(X)

# 2. Límites de la Neuroevolución
MAX_LAYERS = 5              # Profundidad máxima permitida
MAX_NEURONS = 128           # Ancho máximo por capa
POPULATION_SIZE = 15        # Cantidad de individuos en la población
GENERATIONS = 12            # Cantidad de generaciones
MUTATION_RATE = 0.3         # Tasa alta para explorar topologías agresivamente

def create_chromosome():
    """
    Crea una topología aleatoria. 
    Usamos una probabilidad de que una capa sea 0 (inexistente) 
    para fomentar arquitecturas variadas desde el inicio.
    """
    chromosome = []
    for _ in range(MAX_LAYERS):
        if random.random() < 0.3: # 30% de probabilidad de no tener esta capa
            chromosome.append(0)
        else:
            chromosome.append(random.randint(4, MAX_NEURONS))
    return chromosome

def decode_chromosome(chromosome):
    """Convierte el vector genético en una tupla válida para MLPClassifier (ignorando ceros)."""
    architecture = tuple([neurons for neurons in chromosome if neurons > 0])
    # Lo usamos Si mutó tanto que perdió todas las capas, le damos una capa mínima de rescate
    return architecture if len(architecture) > 0 else (4,)

def calculate_fitness(chromosome):
    """Entrena la arquitectura descubierta y penaliza redes excesivamente grandes."""
    architecture = decode_chromosome(chromosome)
    
    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation='relu',
        max_iter=150,
        random_state=42
    )
    
    # Evaluar y entrenar precisión
    accuracy = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy').mean()
    
    # Función de Aptitud con Penalización (Regularización Topológica):
    # Restamos un castigo diminuto por cada neurona y cada capa. 
    # Esto empuja al GA a buscar la red MÁS PEQUEÑA que mantenga el MÁXIMO accuracy.
    total_neurons = sum(architecture)
    total_layers = len(architecture)
    
    alpha_neurons = 0.0001
    alpha_layers = 0.001
    
    fitness = accuracy - (alpha_neurons * total_neurons) - (alpha_layers * total_layers) # formula= accuracy - (0.0001 * total_neurons) - (0.001 * total_layers)
    return fitness, accuracy, architecture

def crossover(parent1, parent2):
    """Cruce de un punto, permitiendo intercambiar bloques enteros de capas ocultas."""
    point = random.randint(1, MAX_LAYERS - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    """Muta alterando drásticamente el número de neuronas o eliminando/creando capas."""
    for i in range(MAX_LAYERS):
        if random.random() < MUTATION_RATE:
            if chromosome[i] > 0 and random.random() < 0.2:
                # 20% de probabilidad de "apagar" esta capa (Drop Layer)
                chromosome[i] = 0
            else:
                # Modificar el tamaño de la capa (puede resucitar una capa apagada)
                chromosome[i] = random.randint(4, MAX_NEURONS)
    return chromosome

# --- BUCLE PRINCIPAL DE NEUROEVOLUCIÓN ---
print("Iniciando Evolución de Arquitecturas (NAS)...")
population = [create_chromosome() for _ in range(POPULATION_SIZE)] # población inicial

best_overall_fitness = -float('inf') # para guardar puntuación total (accuracy + penalización)
best_overall_acc = 0                 # SOLO la precisión
best_architecture = None             # la red (array/tupla)

for generation in range(GENERATIONS):
    evaluated = [calculate_fitness(ind) for ind in population]
    
    # Extraer métricas para limpieza de código
    fitnesses = [eval[0] for eval in evaluated]
    accuracies = [eval[1] for eval in evaluated]
    architectures = [eval[2] for eval in evaluated]
    
    best_gen_idx = np.argmax(fitnesses)
    
    if fitnesses[best_gen_idx] > best_overall_fitness:
        best_overall_fitness = fitnesses[best_gen_idx]
        best_overall_acc = accuracies[best_gen_idx]
        best_architecture = architectures[best_gen_idx]
        best_chromosome = population[best_gen_idx].copy()
        
    print(f"Gen {generation + 1:02d} | Mejor Acc: {accuracies[best_gen_idx]:.4f} | Topología: {architectures[best_gen_idx]}")
    
    # Elitismo
    new_population = [best_chromosome]
    
    while len(new_population) < POPULATION_SIZE:
        # Selección por Torneo
        t1 = random.sample(range(POPULATION_SIZE), 3)
        p1 = population[max(t1, key=lambda i: fitnesses[i])]
        
        t2 = random.sample(range(POPULATION_SIZE), 3)
        p2 = population[max(t2, key=lambda i: fitnesses[i])]
        
        c1, c2 = crossover(p1, p2)
        new_population.extend([mutate(c1), mutate(c2)])
        
    population = new_population[:POPULATION_SIZE]

print("\n" + "="*50)
print("NEUROEVOLUCIÓN COMPLETADA")
print("="*50)
print(f"Mejor Precisión Real : {best_overall_acc:.4f}")
print(f"Topología Óptima     : {best_architecture} ({len(best_architecture)} capas ocultas, {sum(best_architecture)} neuronas totales)")