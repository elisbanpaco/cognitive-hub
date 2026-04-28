import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

class PSOClustering:
    """
    Algoritmo PSO aplicado a problemas de Agrupamiento (Clustering).
    Busca optimizar las coordenadas de K centroides minimizando 
    la distancia intra-cluster (Suma de Errores Cuadráticos - SSE).
    """
    def __init__(self, n_clusters, num_particles=20, max_iter=50, w=0.729, c1=1.494, c2=1.494, random_state=42):
        self.n_clusters = n_clusters
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        # Coeficientes estándar recomendados en la literatura PSO
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        # Estado
        self.gbest_position = None
        self.gbest_fitness = np.inf # Para clustering, buscamos MINIMIZAR el error
        self.labels_ = None

    def _calculate_fitness(self, particle_position, X):
        """
        [Función de Aptitud]
        Calcula la métrica de Cuantización del Error (Suma de distancias euclidianas 
        desde cada punto de X a su centroide más cercano).
        """
        # Reconstruir la matriz de centroides (K, D) a partir del vector plano
        centroids = particle_position.reshape(self.n_clusters, X.shape[1])
        
        # cdist calcula la distancia de cada punto a cada centroide
        distances = cdist(X, centroids, metric='euclidean')
        
        # Obtener la distancia mínima para cada punto (al centroide más cercano)
        min_distances = np.min(distances, axis=1)
        
        # La aptitud es la suma de esos errores (buscamos minimizarla)
        return np.sum(min_distances)

    def fit(self, X):
        n_samples, n_features = X.shape
        self.particle_dim = self.n_clusters * n_features
        
        # Límites del espacio de búsqueda basados en los datos
        bounds_min = np.min(X, axis=0)
        bounds_max = np.max(X, axis=0)
        
        # [Inicialización del Enjambre]
        self.positions = np.zeros((self.num_particles, self.particle_dim))
        
        # Para evitar óptimos locales (debilidad del PSO según la teoría), 
        # inicializamos las partículas seleccionando puntos aleatorios reales del dataset
        for i in range(self.num_particles):
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            initial_centroids = X[random_indices]
            self.positions[i] = initial_centroids.flatten()
            
        # Inicializar velocidades a 0
        self.velocities = np.zeros((self.num_particles, self.particle_dim))
        
        # Memoria cognitiva (pbest)
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(self.num_particles, np.inf)

        print(f"Iniciando PSO Clustering para {self.n_clusters} grupos...")

        # [Evolución del Enjambre]
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # 1. Evaluar aptitud
                fitness = self._calculate_fitness(self.positions[i], X)
                
                # 2. Actualizar mejor personal (Buscando el MENOR error)
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                    
                    # 3. Actualizar mejor global
                    if fitness < self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.positions[i].copy()
                        
            # 4. Cinemática de la partícula (Actualización)
            r1 = np.random.rand(self.num_particles, self.particle_dim)
            r2 = np.random.rand(self.num_particles, self.particle_dim)
            
            cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
            social = self.c2 * r2 * (self.gbest_position - self.positions)
            
            self.velocities = (self.w * self.velocities) + cognitive + social
            self.positions = self.positions + self.velocities
            
            # [Restricción] Mantener los centroides dentro del bounding box de los datos
            for d in range(n_features):
                # Aplicar límites a cada feature en el vector aplanado
                for k in range(self.n_clusters):
                    idx = k * n_features + d
                    self.positions[:, idx] = np.clip(self.positions[:, idx], bounds_min[d], bounds_max[d])

            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteración {iteration + 1:03d}/{self.max_iter} | Error Global (SSE): {self.gbest_fitness:.4f}")

        # [Finalización]
        # Calcular las etiquetas finales para cada dato
        best_centroids = self.gbest_position.reshape(self.n_clusters, n_features)
        distances = cdist(X, best_centroids, metric='euclidean')
        self.labels_ = np.argmin(distances, axis=1)
        
        return best_centroids, self.labels_

# ==========================================
# Ejecución y prueba del algoritmo
# ==========================================
if __name__ == "__main__":
    # Usaremos el dataset Iris (omitiendo las etiquetas reales para hacer clustering ciego)
    data = load_iris()
    X = data.data
    
    # Escalar datos es fundamental basada en distancias euclidianas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3 especies de flores = 3 clusters
    n_clusters = 3 
    
    pso_clustering = PSOClustering(n_clusters=n_clusters, num_particles=30, max_iter=50)
    best_centroids, labels = pso_clustering.fit(X_scaled)

    print("\n--- Resultados de Finalización ---")
    print(f"Error Final de Agrupamiento: {pso_clustering.gbest_fitness:.4f}")
    
    # Opcional: Mostrar cuántos elementos quedaron en cada cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} elementos")