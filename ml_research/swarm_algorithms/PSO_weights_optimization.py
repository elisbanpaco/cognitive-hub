import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

class FeedForwardNN:
    """
    Representación de una Red Neuronal de 3 capas (Entrada, Oculta, Salida)
    diseñada para recibir pesos de forma vectorizada (ideal para PSO).
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Calcular la cantidad total de parámetros (pesos + sesgos)
        self.num_params = (self.input_size * self.hidden_size) + self.hidden_size + \
                          (self.hidden_size * self.output_size) + self.output_size

    def set_weights(self, weight_vector):
        """
        [Representación] Reconstruye las matrices de pesos y vectores de sesgo 
        a partir de un vector plano de una partícula.
        """
        idx = 0
        
        # Pesos y sesgo de la Capa Oculta (W1, b1)
        end = self.input_size * self.hidden_size
        self.W1 = weight_vector[idx:end].reshape((self.input_size, self.hidden_size))
        idx = end
        
        end = idx + self.hidden_size
        self.b1 = weight_vector[idx:end].reshape((1, self.hidden_size))
        idx = end
        
        # Pesos y sesgo de la Capa de Salida (W2, b2)
        end = idx + (self.hidden_size * self.output_size)
        self.W2 = weight_vector[idx:end].reshape((self.hidden_size, self.output_size))
        idx = end
        
        self.b2 = weight_vector[idx:].reshape((1, self.output_size))

    def forward(self, X):
        """Paso hacia adelante (Forward propagation)"""
        # Capa oculta con activación ReLU
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1) 
        
        # Capa de salida con activación Sigmoide (salida entre 0 y 1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = 1 / (1 + np.exp(-np.clip(Z2, -500, 500))) # Clip para evitar overflow
        
        return A2

class PSONeuralNetTrainer:
    """
    Entrenador de Redes Neuronales utilizando PSO (sin backpropagation).
    """
    def __init__(self, nn_model, num_particles=30, max_iter=100, w=0.6, c1=1.5, c2=1.5, bounds=(-2.0, 2.0)):
        self.nn = nn_model
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        
        self.num_dimensions = self.nn.num_params
        
        # Variables de estado del enjambre
        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_fitness = np.full(self.num_particles, -np.inf) # Buscamos maximizar
        self.gbest_position = None
        self.gbest_fitness = -np.inf

    def _calculate_fitness(self, position, X, y):
        """
        [Función de Aptitud] 
        Evaluamos la partícula calculando la pérdida de Entropía Cruzada Binaria (Log Loss).
        Como PSO típicamente maximiza, retornaremos la pérdida en negativo.
        """
        self.nn.set_weights(position)
        y_pred_proba = self.nn.forward(X)
        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15) # Evitar log(0)
        
        # Calculamos Log Loss (Binary Cross-Entropy)
        loss = -np.mean(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba))
        
        # Retornamos el valor negativo para que el PSO maximice la aptitud (minimice el error)
        return -loss

    def fit(self, X, y):
        # [Inicialización del Enjambre]
        self.positions = np.random.uniform(
            low=self.bounds[0], high=self.bounds[1], 
            size=(self.num_particles, self.num_dimensions)
        )
        self.velocities = np.random.uniform(-0.1, 0.1, size=(self.num_particles, self.num_dimensions))
        self.pbest_positions = self.positions.copy()

        print("Iniciando entrenamiento de la red con PSO...")
        
        # [Evolución del Enjambre]
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # 1. Evaluar partícula
                fitness = self._calculate_fitness(self.positions[i], X, y)
                
                # 2. Actualizar mejor personal
                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                    
                    # 3. Actualizar mejor global
                    if fitness > self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.positions[i].copy()

            # 4. Comportamiento de la partícula (Cinemática)
            r1 = np.random.rand(self.num_particles, self.num_dimensions)
            r2 = np.random.rand(self.num_particles, self.num_dimensions)
            
            cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
            social = self.c2 * r2 * (self.gbest_position - self.positions)
            
            self.velocities = (self.w * self.velocities) + cognitive + social
            self.positions = self.positions + self.velocities
            
            # Limitar los pesos para evitar explosión de valores
            self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

            # Monitoreo
            if (iteration + 1) % 10 == 0 or iteration == 0:
                current_loss = -self.gbest_fitness
                # Calcular Accuracy actual con la mejor posición
                self.nn.set_weights(self.gbest_position)
                y_pred = (self.nn.forward(X) > 0.5).astype(int)
                acc = accuracy_score(y, y_pred)
                print(f"Iteración {iteration + 1:03d}/{self.max_iter} | Loss BCE: {current_loss:.4f} | Accuracy: {acc:.4f}")

        # [Finalización]
        self.nn.set_weights(self.gbest_position)
        return self.gbest_position

# ==========================================
# Ejecución y prueba del algoritmo
# ==========================================
if __name__ == "__main__":
    # 1. Preparación de datos
    data = load_breast_cancer()
    X = data.data
    y = data.target.reshape(-1, 1) # Asegurar dimensión de columna

    # CRÍTICO: Escalar los datos es vital para redes neuronales y algoritmos metaheurísticos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]
    hidden_dim = 10 # 10 neuronas en la capa oculta
    output_dim = 1  # 1 neurona de salida (0 o 1)

    # 2. Instanciamos la arquitectura (al menos 3 capas)
    nn = FeedForwardNN(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)
    print(f"Arquitectura: {input_dim} Entradas -> {hidden_dim} Ocultas -> {output_dim} Salida")
    print(f"Total de pesos y sesgos a optimizar (Dimensiones de la partícula): {nn.num_params}\n")

    # 3. Entrenamos usando el Enjambre
    pso_trainer = PSONeuralNetTrainer(nn_model=nn, num_particles=50, max_iter=150)
    best_weights = pso_trainer.fit(X_scaled, y)

    # 4. Evaluación final
    print("\n--- Resultados de Finalización (NN Training with PSO) ---")
    nn.set_weights(best_weights)
    y_pred_final = (nn.forward(X_scaled) > 0.5).astype(int)
    final_accuracy = accuracy_score(y, y_pred_final)
    print(f"Accuracy Final en el dataset de entrenamiento: {final_accuracy:.4f}")