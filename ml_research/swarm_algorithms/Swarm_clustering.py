import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Configuración profesional de logs en lugar de "prints" esparcidos
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class PSOClustering:
    """
    Optimización por Enjambre de Partículas (PSO) para Clustering.
    Optimiza las coordenadas de K centroides minimizando el SSE.
    """
    def __init__(self, n_clusters: int, num_particles: int = 20, max_iter: int = 50, 
                 w: float = 0.729, c1: float = 1.494, c2: float = 1.494, random_state: int = 42):
        self.n_clusters = n_clusters
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        np.random.seed(random_state)
        
        self.gbest_position: np.ndarray = None
        self.gbest_fitness: float = np.inf
        self.convergence_history: List[float] = []

    def _calculate_fitness(self, position: np.ndarray, X: np.ndarray) -> float:
        """Calcula el SSE (Suma de Errores Cuadráticos) para un conjunto de centroides."""
        centroids = position.reshape(self.n_clusters, -1)
        return cdist(X, centroids, metric='euclidean').min(axis=1).sum()

    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, n_features = X.shape
        particle_dim = self.n_clusters * n_features
        
        # 1. Vectorización de límites (Elimina el doble bucle for del código original)
        bounds_min = np.tile(X.min(axis=0), self.n_clusters)
        bounds_max = np.tile(X.max(axis=0), self.n_clusters)
        
        # 2. Inicialización limpia de partículas (comprensión de listas)
        random_indices = [np.random.choice(n_samples, self.n_clusters, replace=False) 
                          for _ in range(self.num_particles)]
        self.positions = np.array([X[idx].flatten() for idx in random_indices])
        self.velocities = np.zeros_like(self.positions)
        
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(self.num_particles, np.inf)

        logging.info(f"Iniciando PSO Clustering para {self.n_clusters} grupos...")

        for iteration in range(self.max_iter):
            # 3. Evaluación de fitness y actualización de PBest usando máscaras booleanas (vectorizado)
            fitness = np.array([self._calculate_fitness(p, X) for p in self.positions])
            
            better_mask = fitness < self.pbest_fitness
            self.pbest_fitness[better_mask] = fitness[better_mask]
            self.pbest_positions[better_mask] = self.positions[better_mask]
            
            # 4. Actualización de GBest
            best_idx = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[best_idx] < self.gbest_fitness:
                self.gbest_fitness = self.pbest_fitness[best_idx]
                self.gbest_position = self.pbest_positions[best_idx].copy()
            
            self.convergence_history.append(self.gbest_fitness)
            
            # 5. Cálculo de cinemática (100% matricial)
            r1 = np.random.rand(self.num_particles, particle_dim)
            r2 = np.random.rand(self.num_particles, particle_dim)
            
            cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
            social = self.c2 * r2 * (self.gbest_position - self.positions)
            
            self.velocities = (self.w * self.velocities) + cognitive + social
            # Clip matricial instantáneo (reemplaza el costoso bucle anidado del código original)
            self.positions = np.clip(self.positions + self.velocities, bounds_min, bounds_max)

            if (iteration + 1) % 10 == 0 or iteration == 0:
                logging.info(f"Iteración {iteration+1:03d}/{self.max_iter} | Error Global (SSE): {self.gbest_fitness:.4f}")

        # Retorno final
        best_centroids = self.gbest_position.reshape(self.n_clusters, n_features)
        labels = cdist(X, best_centroids, metric='euclidean').argmin(axis=1)
        return best_centroids, labels

 # Vizualizacion
def plot_convergence(history: List[float]):
    """Grafica la curva de convergencia."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', color='b', markersize=3)
    plt.title('Curva de Convergencia del PSO')
    plt.xlabel('Iteración')
    plt.ylabel('SSE')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pca_clusters(X_scaled: np.ndarray, centroids: np.ndarray, labels: np.ndarray):
    """Grafica los datos y centroides reducidos a 2D con PCA."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(centroids)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='*', s=200, 
                edgecolors='black', label='Centroides')
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'PSO Clustering ({len(centroids)} grupos) - Proyección PCA 2D')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# MAIN PRINCIPAL
def main():
    df = pd.read_csv("Patient_Health_Risk.csv")

    X = df.drop(columns=['Class']).values
    y = df['Class'].values
    n_clusters = len(np.unique(y))
    logging.info(f"Detectadas {n_clusters} clases objetivo.")
    
    # Preprocesamiento
    X_scaled = StandardScaler().fit_transform(X)
    
    # Modelo y Entrenamiento
    pso = PSOClustering(n_clusters=n_clusters, num_particles=30, max_iter=50)
    best_centroids, labels = pso.fit(X_scaled)
    
    # Resultados
    logging.info(f"SSE Final: {pso.gbest_fitness:.4f}")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        logging.info(f"Cluster {label}: {count} elementos")
    
    # Visualizaciones
    plot_convergence(pso.convergence_history)
    plot_pca_clusters(X_scaled, best_centroids, labels)

if __name__ == "__main__":
    main()