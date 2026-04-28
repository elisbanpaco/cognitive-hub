import numpy as np
import random
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore") # para gnorar warnings de convergencia para mantener la consola limpia

# 1. Preparar el Dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Las redes neuronales son muy sensibles a la escala de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset escalado y listo. Forma de X: {X.shape}")

# 2. Definir el Espacio de Búsqueda (Search Space)
# Aquí es donde ocurre la "magia" del diseño
SEARCH_SPACE = {
    'hidden_layer_sizes': [(16,), (32,), (64,), (32, 16), (64, 32)],
    'activation': ['tanh', 'relu', 'logistic'],
    'alpha': (0.0001, 0.1),            # Rango continuo
    'learning_rate_init': (0.001, 0.1) # Rango continuo
}

# 3. Hiperparámetros del Algoritmo Genético
POPULATION_SIZE = 10
GENERATIONS = 10
MUTATION_RATE = 0.2 # Tasa un poco más alta para evitar estancamiento

def create_individual():
    """Crea un individuo aleatorio basado en el espacio de búsqueda."""
    return {
        'hidden_layer_sizes': random.choice(SEARCH_SPACE['hidden_layer_sizes']),
        'activation': random.choice(SEARCH_SPACE['activation']),
        'alpha': random.uniform(SEARCH_SPACE['alpha'][0], SEARCH_SPACE['alpha'][1]),
        'learning_rate_init': random.uniform(SEARCH_SPACE['learning_rate_init'][0], SEARCH_SPACE['learning_rate_init'][1])
    }

def calculate_fitness(individual):
    """Evalúa la configuración de hiperparámetros entrenando el modelo."""
    # Instanciar la red neuronal con los genes del individuo
    model = MLPClassifier(
        hidden_layer_sizes=individual['hidden_layer_sizes'],
        activation=individual['activation'],
        alpha=individual['alpha'],
        learning_rate_init=individual['learning_rate_init'],
        max_iter=100, # Límite bajo para que el ejemplo corra rápido
        random_state=42
    )
    
    # Validar con Cross-Validation
    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
    return scores.mean()

def tournament_selection(population, fitnesses, k=3):
    """Selección por torneo."""
    selected_indices = random.sample(range(len(population)), k)
    best_index = max(selected_indices, key=lambda i: fitnesses[i])
    return population[best_index].copy()

def crossover(parent1, parent2):
    """Cruce Uniforme: para cada gen, elige aleatoriamente de uno de los padres."""
    child1 = {}
    child2 = {}
    
    for key in parent1.keys():
        if random.random() < 0.5:
            child1[key] = parent1[key]
            child2[key] = parent2[key]
        else:
            child1[key] = parent2[key]
            child2[key] = parent1[key]
            
    return child1, child2

def mutate(individual):
    """Muta un gen aleatorio dentro de sus límites permitidos."""
    if random.random() < MUTATION_RATE:
        # Elegir un gen al azar para mutar
        gene_to_mutate = random.choice(list(individual.keys()))
        
        if gene_to_mutate == 'hidden_layer_sizes':
            individual[gene_to_mutate] = random.choice(SEARCH_SPACE['hidden_layer_sizes'])
        elif gene_to_mutate == 'activation':
            individual[gene_to_mutate] = random.choice(SEARCH_SPACE['activation'])
        elif gene_to_mutate == 'alpha':
            individual[gene_to_mutate] = random.uniform(SEARCH_SPACE['alpha'][0], SEARCH_SPACE['alpha'][1])
        elif gene_to_mutate == 'learning_rate_init':
            individual[gene_to_mutate] = random.uniform(SEARCH_SPACE['learning_rate_init'][0], SEARCH_SPACE['learning_rate_init'][1])
            
    return individual

# --- BUCLE PRINCIPAL ---
print("\nIniciando Búsqueda de Arquitectura y Parámetros (NAS/HPO)...")
population = [create_individual() for _ in range(POPULATION_SIZE)]

best_overall_ind = None
best_overall_fitness = 0.0

for generation in range(GENERATIONS):
    fitnesses = [calculate_fitness(ind) for ind in population]
    
    best_gen_fitness = max(fitnesses)
    best_gen_idx = fitnesses.index(best_gen_fitness)
    best_gen_ind = population[best_gen_idx]
    
    if best_gen_fitness > best_overall_fitness:
        best_overall_fitness = best_gen_fitness
        best_overall_ind = best_gen_ind.copy()
        
    print(f"Generación {generation + 1:02d} | Mejor Accuracy: {best_gen_fitness:.4f}")
    
    new_population = [best_gen_ind] # Elitismo
    
    while len(new_population) < POPULATION_SIZE:
        p1 = tournament_selection(population, fitnesses)
        p2 = tournament_selection(population, fitnesses)
        
        c1, c2 = crossover(p1, p2)
        
        c1 = mutate(c1)
        c2 = mutate(c2)
        
        new_population.extend([c1, c2])
        
    population = new_population[:POPULATION_SIZE]

# --- RESULTADOS FINALES ---
print("\n" + "="*50)
print("OPTIMIZACIÓN COMPLETADA")
print("="*50)
print(f"Mejor Accuracy global : {best_overall_fitness:.4f}")
print("Mejor Arquitectura / Hiperparámetros encontrados:")
for key, value in best_overall_ind.items():
    if isinstance(value, float):
        print(f"  - {key}: {value:.6f}")
    else:
        print(f"  - {key}: {value}")