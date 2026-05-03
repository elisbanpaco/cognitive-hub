import warnings
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Ocultar advertencias de Sklearn para mantener limpia la salida estándar
warnings.filterwarnings('ignore')

class PSOHyperparameterTuning:
    """
    Particle Swarm Optimization (PSO) algorithm for SVM hyperparameter tuning.
    Optimizes 4 parameters: C, gamma, epsilon, and tol.
    """

    def __init__(
        self, 
        bounds: List[List[float]], 
        num_particles: int = 15, 
        max_iter: int = 20, 
        w: float = 0.5, 
        c1: float = 1.5, 
        c2: float = 1.5, 
        random_state: int = 42
    ) -> None:
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = np.array(bounds)
        
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        np.random.seed(random_state)
        
        # Swarm state
        self.positions: np.ndarray = None
        self.velocities: np.ndarray = None
        
        # Cognitive memory
        self.pbest_positions: np.ndarray = None
        self.pbest_fitness: np.ndarray = np.full(self.num_particles, -1.0)
        
        # Social memory
        self.gbest_position: np.ndarray = None
        self.gbest_fitness: float = -1.0
        
        self.fitness_history: List[float] = []

    def _calculate_fitness(self, position: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluates the fitness of a particle's position using 3-fold CV."""
        # Nota: Epsilon se desempaqueta pero no se usa en la instanciación estándar de SVC. 
        # Se utiliza '_' para indicar explícitamente que es una variable ignorada.
        C_val, gamma_val, _, tol_val = position
        
        model = SVC(
            C=C_val, 
            gamma=gamma_val, 
            kernel='rbf',
            tol=tol_val,
            max_iter=1000,
            cache_size=200
        )
        
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Runs the PSO optimization loop."""
        num_dimensions = self.bounds.shape[0]
        
        self.positions = np.random.uniform(
            low=self.bounds[:, 0], 
            high=self.bounds[:, 1], 
            size=(self.num_particles, num_dimensions)
        )
        
        self.velocities = np.random.uniform(
            low=-0.1, high=0.1, 
            size=(self.num_particles, num_dimensions)
        )
        
        self.pbest_positions = self.positions.copy()
        
        # Initial evaluation
        for i in range(self.num_particles):
            fitness = self._calculate_fitness(self.positions[i], X, y)
            self.pbest_fitness[i] = fitness
            if fitness > self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = self.positions[i].copy()

        print("----------------------------------------------------------------------")
        print(f"Starting PSO Optimization | Particles: {self.num_particles} | Iterations: {self.max_iter}")
        print("Parameters: C, gamma, epsilon, tol")
        print("----------------------------------------------------------------------")
        
        # Main evolution loop
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                fitness = self._calculate_fitness(self.positions[i], X, y)
                
                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                    
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()

            for i in range(self.num_particles):
                r1 = np.random.rand(num_dimensions)
                r2 = np.random.rand(num_dimensions)
                
                cognitive_velocity = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.gbest_position - self.positions[i])
                
                self.velocities[i] = (self.w * self.velocities[i]) + cognitive_velocity + social_velocity
                self.positions[i] = self.positions[i] + self.velocities[i]
                
                # Boundary constraints
                self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])
            
            self.fitness_history.append(self.gbest_fitness)
            
            print(f"Iter {iteration+1:2d}/{self.max_iter} | Best Accuracy: {self.gbest_fitness:.4f} | "
                  f"C: {self.gbest_position[0]:6.2f} | γ: {self.gbest_position[1]:.6f} | "
                  f"ε: {self.gbest_position[2]:.5f} | tol: {self.gbest_position[3]:.5f}")

        print("Optimization completed.")
        return self.gbest_position, self.gbest_fitness


def plot_convergence(history: List[float]) -> None:
    """Plots the fitness history convergence over iterations."""
    plt.figure(figsize=(8, 5))
    plt.plot(history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy (Fitness)', fontsize=12)
    plt.title('PSO Convergence Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.0)
    
    for i, (iter_num, fitness) in enumerate(zip(range(1, len(history)+1), history)):
        if i in (0, 3, len(history)-1):
            plt.annotate(f'{fitness:.3f}', (iter_num-1, fitness), 
                         textcoords="offset points", xytext=(5,5), fontsize=9)
    plt.show()


def main() -> None:
    print("\n" + "=" * 70)
    print("PSO HYPERPARAMETER OPTIMIZATION (4 PARAMETERS)")
    print("=" * 70)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print("\nDataset Information:")
    print(f"  Name: Breast Cancer Wisconsin")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: 2 (Malignant=0, Benign=1)")
    
    bounds = [
        [0.1, 100.0],      # C
        [0.0001, 2.0],     # gamma
        [0.01, 1.0],       # epsilon
        [0.00001, 0.1]     # tol
    ]
    
    print("\nEvaluating Baseline (Default SVM Parameters)...")
    baseline_model = SVC(kernel='rbf')
    baseline_score = cross_val_score(baseline_model, X, y, cv=3, scoring='accuracy').mean()
    print(f"Baseline Accuracy: {baseline_score:.4f}")
    print("-" * 70)
    
    pso = PSOHyperparameterTuning(
        bounds=bounds, 
        num_particles=15, 
        max_iter=15,
        w=0.7,
        c1=1.5,
        c2=1.5,
        random_state=42
    )
    
    best_params, best_accuracy = pso.fit(X, y)
    plot_convergence(pso.fitness_history)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Baseline Accuracy: {baseline_score:.4f}")
    print(f"Optimized Accuracy: {best_accuracy:.4f}")
    print(f"Improvement: {((best_accuracy - baseline_score) / baseline_score) * 100:.2f}%\n")
    
    print("Best Hyperparameters:")
    print(f"  C:       {best_params[0]:.4f}")
    print(f"  gamma:   {best_params[1]:.6f}")
    print(f"  epsilon: {best_params[2]:.6f}")
    print(f"  tol:     {best_params[3]:.6f}\n")
    
    print("Running final cross-validation with optimized parameters...")
    best_model = SVC(
        C=best_params[0], 
        gamma=best_params[1], 
        kernel='rbf', 
        tol=best_params[3], 
        max_iter=1000
    )
    final_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f"Final 5-fold CV Accuracy: {final_scores.mean():.4f} ± {final_scores.std():.4f}")


if __name__ == "__main__":
    main()