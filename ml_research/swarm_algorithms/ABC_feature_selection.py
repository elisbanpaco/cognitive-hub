import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import warnings

# Ocultamos advertencias para mantener la salida de consola limpia
warnings.filterwarnings('ignore')

class ABCFeatureSelection:
    """
    Algoritmo de Colonia Artificial de Abejas (ABC) para Feature Selection.
    """
    def __init__(self, num_bees=30, max_iter=50, limit=20, random_state=42):
        self.num_bees = num_bees
        self.num_foods = num_bees // 2  # La mitad son obreras, la mitad observadoras
        self.max_iter = max_iter
        self.limit = limit # Límite de intentos antes de que una fuente se agote (Abejas exploradoras)
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.food_sources = None
        self.fitness = None
        self.trial_counters = None
        
        self.best_solution = None
        self.best_fitness = -1.0
        self.best_binary_mask = None

    def _get_binary_mask(self, continuous_vector):
        """
        [Representación de la partícula]
        Convierte el vector continuo en el rango [0, 1] a un vector binario.
        """
        return (continuous_vector > 0.5).astype(int)

    def _calculate_fitness(self, continuous_vector, X, y):
        """
        [Función de aptitud]
        Evalúa la calidad de las características seleccionadas entrenando un modelo.
        Maximizamos el Accuracy de un K-Nearest Neighbors usando validación cruzada.
        """
        binary_mask = self._get_binary_mask(continuous_vector)
        
        # Penalizar si no se selecciona ninguna característica
        if np.sum(binary_mask) == 0:
            return 0.0
            
        X_subset = X[:, binary_mask == 1]
        
        # Usamos KNN como modelo evaluador rápido
        clf = KNeighborsClassifier(n_neighbors=5)
        # Obtenemos el accuracy medio mediante validación cruzada (3 folds)
        scores = cross_val_score(clf, X_subset, y, cv=3, scoring='accuracy')
        
        return scores.mean()

    def _mutate(self, current_food, partner_food):
        """
        Ecuación de mutación estándar del algoritmo ABC.
        v_{i,j} = x_{i,j} + phi * (x_{i,j} - x_{k,j})
        """
        phi = np.random.uniform(-1, 1, size=current_food.shape)
        new_food = current_food + phi * (current_food - partner_food)
        # Mantener los valores dentro del espacio de búsqueda válido [0, 1]
        return np.clip(new_food, 0.0, 1.0)

    def fit(self, X, y):
        num_features = X.shape[1]
        
        # [Inicialización del enjambre]
        self.food_sources = np.random.uniform(0, 1, (self.num_foods, num_features))
        self.fitness = np.zeros(self.num_foods)
        self.trial_counters = np.zeros(self.num_foods)
        
        # Evaluación inicial
        for i in range(self.num_foods):
            self.fitness[i] = self._calculate_fitness(self.food_sources[i], X, y)
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.food_sources[i].copy()
                self.best_binary_mask = self._get_binary_mask(self.best_solution)

        # [Evolución del enjambre] - Ciclo Principal
        for iteration in range(self.max_iter):
            
            # --- Fase 1: Abejas Obreras (Employed Bees) ---
            # Explotan las fuentes de alimento actuales
            for i in range(self.num_foods):
                # Seleccionar una pareja aleatoria distinta a la actual
                partner_idx = np.random.choice([idx for idx in range(self.num_foods) if idx != i])
                
                # Generar nueva solución (vecindad)
                new_food = self._mutate(self.food_sources[i], self.food_sources[partner_idx])
                new_fitness = self._calculate_fitness(new_food, X, y)
                
                # Selección codiciosa (Greedy selection)
                if new_fitness > self.fitness[i]:
                    self.food_sources[i] = new_food
                    self.fitness[i] = new_fitness
                    self.trial_counters[i] = 0
                else:
                    self.trial_counters[i] += 1

            # --- Fase 2: Abejas Observadoras (Onlooker Bees) ---
            # Seleccionan fuentes de alimento basadas en su probabilidad (aptitud)
            # Normalizamos la aptitud para obtener probabilidades
            fitness_sum = np.sum(self.fitness)
            if fitness_sum == 0:
                probabilities = np.ones(self.num_foods) / self.num_foods
            else:
                probabilities = self.fitness / fitness_sum
                
            t = 0
            i = 0
            while t < self.num_foods:
                # Selección por ruleta
                if np.random.rand() < probabilities[i]:
                    t += 1
                    partner_idx = np.random.choice([idx for idx in range(self.num_foods) if idx != i])
                    
                    new_food = self._mutate(self.food_sources[i], self.food_sources[partner_idx])
                    new_fitness = self._calculate_fitness(new_food, X, y)
                    
                    if new_fitness > self.fitness[i]:
                        self.food_sources[i] = new_food
                        self.fitness[i] = new_fitness
                        self.trial_counters[i] = 0
                    else:
                        self.trial_counters[i] += 1
                i = (i + 1) % self.num_foods

            # Actualizamos la mejor solución global encontrada hasta el momento
            best_iter_idx = np.argmax(self.fitness)
            if self.fitness[best_iter_idx] > self.best_fitness:
                self.best_fitness = self.fitness[best_iter_idx]
                self.best_solution = self.food_sources[best_iter_idx].copy()
                self.best_binary_mask = self._get_binary_mask(self.best_solution)

            # --- Fase 3: Abejas Exploradoras (Scout Bees) ---
            # Si una fuente se agota (supera el límite de intentos), se busca una nueva aleatoria
            for i in range(self.num_foods):
                if self.trial_counters[i] >= self.limit:
                    self.food_sources[i] = np.random.uniform(0, 1, num_features)
                    self.fitness[i] = self._calculate_fitness(self.food_sources[i], X, y)
                    self.trial_counters[i] = 0

            print(f"Iteración {iteration + 1}/{self.max_iter} | Mejor Accuracy (Aptitud): {self.best_fitness:.4f} | Features usadas: {np.sum(self.best_binary_mask)}")

        # [Finalización]
        return self.best_binary_mask, self.best_fitness

# ==========================================
# Ejecución y prueba del algoritmo
# ==========================================
if __name__ == "__main__":
    # Cargamos un dataset real (Breast Cancer de sklearn)
    print("Cargando dataset 'Breast Cancer'...")
    data = load_breast_cancer()
    X = data.data    # 30 características originales
    y = data.target

    print(f"Dimensiones originales: {X.shape[1]} características.")
    
    # Baseline: Accuracy con todas las características
    baseline_clf = KNeighborsClassifier(n_neighbors=5)
    baseline_score = cross_val_score(baseline_clf, X, y, cv=3, scoring='accuracy').mean()
    print(f"Accuracy sin Feature Selection (Baseline): {baseline_score:.4f}\n")

    # Ejecución del algoritmo ABC
    print("Iniciando optimización por Colonia de Abejas Artificiales (ABC)...")
    abc_optimizer = ABCFeatureSelection(num_bees=20, max_iter=30, limit=10)
    best_features, best_acc = abc_optimizer.fit(X, y)

    print("\n--- Resultados de Finalización ---")
    print(f"Mejor Accuracy obtenido: {best_acc:.4f}")
    print(f"Total de características seleccionadas: {np.sum(best_features)} de {X.shape[1]}")
    
    # Mostrar exactamente qué características fueron seleccionadas
    feature_names = data.feature_names[best_features == 1]
    print(f"Características clave elegidas:\n{feature_names}")