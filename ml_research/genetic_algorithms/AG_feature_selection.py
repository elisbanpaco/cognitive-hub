import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1. Cargar el Dataset (>20 columnas)
data = load_breast_cancer()
X, y = data.data, data.target
N_FEATURES = X.shape[1] # 30 características

print(f"Dataset cargado. Forma de X: {X.shape}")

# 2. Hiperparámetros del Algoritmo Genético
POPULATION_SIZE = 20    # Número de individuos por generación
GENERATIONS = 15        # Número de ciclos de evolución
MUTATION_RATE = 0.05    # 5% de probabilidad de mutar un gen
TOURNAMENT_SIZE = 3     # Individuos compitiendo en la selección

# Inicializar modelo ligero para la evaluación de aptitud
clf = RandomForestClassifier(n_estimators=20, random_state=42)

def calculate_fitness(chromosome):
    """
    Evalúa un individuo entrenando un modelo solo con las características donde el gen es 1.
    Retorna el accuracy medio usando validación cruzada.
    """
    # Obtener los índices donde el cromosoma tiene un 1
    selected_indices = np.where(chromosome == 1)[0]
    
    # Penalizar si el algoritmo elimina TODAS las características
    if len(selected_indices) == 0:
        return 0.0
    
    # Filtrar el dataset original
    X_subset = X[:, selected_indices]
    
    # Evaluar con Cross Validation (3-folds para mayor velocidad)
    scores = cross_val_score(clf, X_subset, y, cv=3, scoring='accuracy')
    
    # Retornamos el accuracy. Podrías modificar esto para que también 
    # penalice (reste puntos) por tener demasiadas características seleccionadas.
    return scores.mean()

def init_population(pop_size, n_features):
    """Genera una población inicial aleatoria de vectores binarios."""
    return [np.random.randint(0, 2, n_features) for _ in range(pop_size)]

def tournament_selection(population, fitnesses):
    """Selecciona un padre haciendo competir a individuos aleatorios."""
    selected_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best_index = max(selected_indices, key=lambda i: fitnesses[i])
    return population[best_index]

def crossover(parent1, parent2):
    """Cruce de un solo punto."""
    if random.random() < 0.8: # 80% de probabilidad de cruce
        point = random.randint(1, N_FEATURES - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(chromosome):
    """Invierte bits al azar basado en la tasa de mutación."""
    for i in range(N_FEATURES):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i] # Cambia 0 a 1 o 1 a 0
    return chromosome

# --- BUCLE PRINCIPAL DE EVOLUCIÓN ---
print("\nIniciando Algoritmo Genético...")
population = init_population(POPULATION_SIZE, N_FEATURES)

best_overall_chromosome = None
best_overall_fitness = 0.0

for generation in range(GENERATIONS):
    # Evaluar la población actual
    fitnesses = [calculate_fitness(ind) for ind in population]
    
    # Registrar al mejor de esta generación
    best_gen_fitness = max(fitnesses)
    best_gen_idx = fitnesses.index(best_gen_fitness)
    best_gen_chromosome = population[best_gen_idx]
    
    if best_gen_fitness > best_overall_fitness:
        best_overall_fitness = best_gen_fitness
        best_overall_chromosome = best_gen_chromosome
        
    num_features_selected = np.sum(best_gen_chromosome)
    
    print(f"Generación {generation + 1:02d} | Mejor Fitness: {best_gen_fitness:.4f} | Características activas: {num_features_selected}/{N_FEATURES}")
    
    # Crear la nueva generación
    new_population = []
    
    # Elitismo: Pasar directamente al mejor individuo a la siguiente generación
    new_population.append(best_gen_chromosome)
    
    while len(new_population) < POPULATION_SIZE:
        # 1. Selección
        p1 = tournament_selection(population, fitnesses)
        p2 = tournament_selection(population, fitnesses)
        
        # 2. Cruce
        c1, c2 = crossover(p1, p2)
        
        # 3. Mutación
        c1 = mutate(c1)
        c2 = mutate(c2)
        
        new_population.extend([c1, c2])
        
    # Asegurar el tamaño exacto de la población (por si agregamos de más en el ciclo)
    population = new_population[:POPULATION_SIZE]

# --- RESULTADOS FINALES ---
final_features = np.where(best_overall_chromosome == 1)[0]
print("\n" + "="*50)
print("EVOLUCIÓN COMPLETADA")
print("="*50)
print(f"Mejor Accuracy global : {best_overall_fitness:.4f}")
print(f"Total características originales : {N_FEATURES}")
print(f"Total características óptimas    : {len(final_features)}")
print(f"Índices de las características   : {final_features}")
print("Nombres de las características   :")
for idx in final_features:
    print(f"  - {data.feature_names[idx]}")