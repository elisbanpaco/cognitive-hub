import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import warnings

# Ocultamos advertencias para mantener la salida limpia
warnings.filterwarnings('ignore')

class PSOHyperparameterTuning:
    """
    Algoritmo de Optimización de Enjambre de Partículas (PSO) 
    para la búsqueda de hiperparámetros de un modelo SVM.
    """
    def __init__(self, bounds, num_particles=15, max_iter=20, w=0.5, c1=1.5, c2=1.5, random_state=42):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = np.array(bounds) # Límites: [[min_C, max_C], [min_gamma, max_gamma]]
        
        # Parámetros físicos del PSO
        self.w = w    # Inercia (qué tanto mantiene su dirección actual)
        self.c1 = c1  # Componente cognitivo (atracción a su mejor posición histórica)
        self.c2 = c2  # Componente social (atracción a la mejor posición de todo el enjambre)
        
        np.random.seed(random_state)
        
        # Estado del enjambre
        self.positions = None
        self.velocities = None
        
        # Memoria individual (Cognitiva)
        self.pbest_positions = None
        self.pbest_fitness = np.full(self.num_particles, -1.0)
        
        # Memoria global (Social)
        self.gbest_position = None
        self.gbest_fitness = -1.0

    def _calculate_fitness(self, position, X, y):
        """
        [Función de aptitud]
        Evalúa los hiperparámetros (posición) entrenando un SVM y 
        retornando el Accuracy promedio por Cross-Validation.
        """
        C_val, gamma_val = position
        
        # Instanciamos el modelo con los hiperparámetros de la partícula
        model = SVC(C=C_val, gamma=gamma_val, kernel='rbf')
        
        # Evaluamos usando validación cruzada (3 folds para agilizar)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()

    def fit(self, X, y):
        num_dimensions = self.bounds.shape[0] # 2 dimensiones: C y Gamma
        
        # [Inicialización del enjambre]
        # Generamos posiciones aleatorias dentro de los límites establecidos
        self.positions = np.random.uniform(
            low=self.bounds[:, 0], 
            high=self.bounds[:, 1], 
            size=(self.num_particles, num_dimensions)
        )
        # Inicializamos las velocidades a 0 (o a un valor pequeño aleatorio)
        self.velocities = np.zeros((self.num_particles, num_dimensions))
        
        # Inicializamos las mejores posiciones personales con las iniciales
        self.pbest_positions = self.positions.copy()

        print("Iniciando evolución del enjambre PSO...")
        
        # [Evolución del enjambre] - Ciclo Principal
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # 1. Evaluar aptitud
                fitness = self._calculate_fitness(self.positions[i], X, y)
                
                # 2. Actualizar mejor posición personal (pbest)
                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                    
                    # 3. Actualizar mejor posición global (gbest)
                    if fitness > self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.positions[i].copy()

            # 4. Comportamiento de la partícula (Actualizar Velocidad y Posición)
            for i in range(self.num_particles):
                # Factores estocásticos (aleatoriedad en cada paso)
                r1 = np.random.rand(num_dimensions)
                r2 = np.random.rand(num_dimensions)
                
                # Ecuación de Velocidad de PSO
                cognitive_velocity = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.gbest_position - self.positions[i])
                
                self.velocities[i] = (self.w * self.velocities[i]) + cognitive_velocity + social_velocity
                
                # Ecuación de Posición de PSO
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # [Restricción] Asegurar que las partículas no se salgan de los límites de búsqueda
                self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])

            print(f"Iteración {iteration + 1}/{self.max_iter} | Mejor Accuracy: {self.gbest_fitness:.4f} | C: {self.gbest_position[0]:.2f}, Gamma: {self.gbest_position[1]:.4f}")

        # [Finalización]
        return self.gbest_position, self.gbest_fitness

# ==========================================
# Ejecución y prueba del algoritmo
# ==========================================
if __name__ == "__main__":
    # Cargamos los datos
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Definimos el espacio de búsqueda (Límites)
    # Dim 1: C (Penalización) -> de 0.1 a 100.0
    # Dim 2: Gamma (Coeficiente Kernel) -> de 0.0001 a 1.0
    bounds = [
        [0.1, 100.0],  # Límites para C
        [0.0001, 1.0]  # Límites para Gamma
    ]

    # Baseline: SVM con hiperparámetros por defecto de Sklearn
    baseline_model = SVC(kernel='rbf')
    baseline_score = cross_val_score(baseline_model, X, y, cv=3, scoring='accuracy').mean()
    print(f"Accuracy de SVM con hiperparámetros por defecto: {baseline_score:.4f}\n")

    # Optimizador PSO
    pso = PSOHyperparameterTuning(bounds=bounds, num_particles=15, max_iter=15)
    best_params, best_accuracy = pso.fit(X, y)

    print("\n--- Resultados de Finalización (Hyperparameter Tuning) ---")
    print(f"Mejor Accuracy obtenido: {best_accuracy:.4f}")
    print(f"Mejores Hiperparámetros -> C: {best_params[0]:.4f}, Gamma: {best_params[1]:.6f}")