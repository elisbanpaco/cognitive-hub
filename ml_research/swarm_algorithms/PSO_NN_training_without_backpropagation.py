import numpy as np
import gymnasium as gym
import pyswarms as ps
from typing import Tuple
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history
class CartPoleNN:
    """Red Neuronal Multicapa (MLP) simple para resolver CartPole."""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8, output_dim: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Total de pesos y sesgos (bias) a optimizar por partícula
        self.num_params = (input_dim * hidden_dim) + hidden_dim + (hidden_dim * output_dim) + output_dim

    def _unpack_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convierte el vector 1D de la partícula en las matrices de pesos de la red."""
        idx = 0
        
        # Capa 1: Pesos y Bias
        w1_end = self.input_dim * self.hidden_dim
        W1 = weights[idx : idx + w1_end].reshape(self.input_dim, self.hidden_dim)
        idx += w1_end
        
        b1_end = self.hidden_dim
        b1 = weights[idx : idx + b1_end]
        idx += b1_end
        
        # Capa 2: Pesos y Bias
        w2_end = self.hidden_dim * self.output_dim
        W2 = weights[idx : idx + w2_end].reshape(self.hidden_dim, self.output_dim)
        idx += w2_end
        
        b2 = weights[idx :]
        
        return W1, b1, W2, b2

    def forward(self, x: np.ndarray, weights: np.ndarray) -> int:
        """Pase hacia adelante (forward pass) sin usar cálculo de gradientes."""
        W1, b1, W2, b2 = self._unpack_weights(weights)
        
        # Capa oculta con activación ReLU
        z1 = np.dot(x, W1) + b1
        a1 = np.maximum(0, z1) 
        
        # Capa de salida (Logits)
        z2 = np.dot(a1, W2) + b2
        
        # Selección de acción (argmax)
        return int(np.argmax(z2))


class Evaluator:
    """Maneja la evaluación del enjambre dentro del entorno de simulación."""
    
    def __init__(self, env_name: str = "CartPole-v1", episodes: int = 3):
        self.env_name = env_name
        self.episodes = episodes  # Promediamos sobre varios episodios para evitar "suerte"
        self.nn = CartPoleNN()

    def evaluate_particle(self, weights: np.ndarray) -> float:
        """Juega un número de episodios con los pesos de una partícula y retorna el costo."""
        env = gym.make(self.env_name)
        total_reward = 0.0
        
        for _ in range(self.episodes):
            state, _ = env.reset()
            done, truncated = False, False
            
            while not (done or truncated):
                action = self.nn.forward(state, weights)
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                
        env.close()
        
        # Promedio de recompensa. PySwarms minimiza la función, así que invertimos el signo.
        # Una recompensa perfecta de 500 será un costo de -500.
        avg_reward = total_reward / self.episodes
        return -avg_reward

    def fitness_function(self, particles: np.ndarray) -> np.ndarray:
        """Evalúa a todo el enjambre. Requerido por la interfaz de pyswarms."""
        n_particles = particles.shape[0]
        costs = np.zeros(n_particles)
        
        for i in range(n_particles):
            costs[i] = self.evaluate_particle(particles[i])
            
        return costs

    def render_best_policy(self, best_weights: np.ndarray):
        """Muestra visualmente a la red neuronal controlando el péndulo."""
        print("\n[INFO] Renderizando la mejor política encontrada...")
        env = gym.make(self.env_name, render_mode="human")
        state, _ = env.reset()
        done, truncated = False, False
        total_reward = 0
        
        while not (done or truncated):
            action = self.nn.forward(state, best_weights)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
        env.close()
        print(f"[INFO] Recompensa final en modo visual: {total_reward}")


def main():
    # 1. Instanciar el evaluador del entorno
    evaluator = Evaluator(episodes=3)
    
    # 2. Configurar hiperparámetros del PSO
    # c1 (cognitivo), c2 (social), w (inercia)
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
    
    # Límites para el espacio de búsqueda continuo (pesos entre -2.0 y 2.0)
    dimensions = evaluator.nn.num_params
    max_bounds = np.ones(dimensions) * 2.0
    min_bounds = np.ones(dimensions) * -2.0
    bounds = (min_bounds, max_bounds)
    
    print(f"[INFO] Iniciando PSO. Partículas explorando en {dimensions} dimensiones continuas...")
    
    # 3. Inicializar el Enjambre (GlobalBestPSO)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=50, 
        dimensions=dimensions, 
        options=options,
        bounds=bounds
    )
    
    # 4. Iniciar el entrenamiento
    # Iteramos 50 veces (generaciones). El costo objetivo ideal es cercano a -500.
    best_cost, best_weights = optimizer.optimize(
        evaluator.fitness_function, 
        iters=50, 
        verbose=True
    )
    
    print("\n[RESULTADOS]")
    print(f"Mejor Recompensa Promedio Alcanzada: {-best_cost} / 500.0")


    print("\n[INFO] Generando gráfica de convergencia...")
    # optimizer.cost_history almacena el mejor costo encontrado en cada iteración
    plot_cost_history(cost_history=optimizer.cost_history)
    plt.title("Convergencia del Enjambre (PSO) en CartPole")
    plt.xlabel("Iteración")
    plt.ylabel("Costo (-Recompensa)")
    plt.show()
    
    # 5. Renderizar el resultado visual del mejor vector de pesos encontrado
    evaluator.render_best_policy(best_weights)

if __name__ == "__main__":
    main()