import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import warnings

# Ocultamos advertencias para mantener la salida limpia
warnings.filterwarnings('ignore')

class PSOHyperparameterTuning:
 
    # Algoritmo de Optimización de Enjambre de Partículas (PSO) para la búsqueda de hiperparámetros de un modelo SVM. Optimiza 4 parámetros: C, gamma, epsilon, tol
    def __init__(self, bounds, num_particles=15, max_iter=20, w=0.5, c1=1.5, c2=1.5, random_state=42):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = np.array(bounds) # Límites: [[min_C, max_C], [min_gamma, max_gamma], [min_epsilon, max_epsilon], [min_tol, max_tol]]
        
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
        
        # Historial de convergencia
        self.fitness_history = []

    def _calculate_fitness(self, position, X, y):

        C_val, gamma_val, epsilon_val, tol_val = position
        
        # Instanciamos el modelo con los 4 hiperparámetros 
        model = SVC(
            C=C_val, 
            gamma=gamma_val, 
            kernel='rbf',
            tol=tol_val,
            max_iter=1000,  # Límite de iteraciones para evitar entrenamientos infinitos
            cache_size=200  # Tamaño de caché para acelerar
        )
        
        # Evaluamos usando validación cruzada (3 folds para agilizar)
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return scores.mean()

    def fit(self, X, y):
        num_dimensions = self.bounds.shape[0] # 4 dimensiones: C, Gamma, Epsilon, Tol
        
        # [Inicialización del enjambre] - Generamos posiciones aleatorias dentro de los límites establecidos
        self.positions = np.random.uniform(
            low=self.bounds[:, 0], 
            high=self.bounds[:, 1], 
            size=(self.num_particles, num_dimensions)
        )
        # Inicializamos las velocidades a un valor pequeño aleatorio (mejor que cero)
        self.velocities = np.random.uniform(
            low=-0.1, high=0.1, 
            size=(self.num_particles, num_dimensions)
        )
        
        # Inicializamos las mejores posiciones personales con las iniciales
        self.pbest_positions = self.positions.copy()
        
        # Evaluación inicial para establecer el mejor global
        for i in range(self.num_particles):
            fitness = self._calculate_fitness(self.positions[i], X, y)
            self.pbest_fitness[i] = fitness
            if fitness > self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = self.positions[i].copy()

        print("=" * 70)
        print("Iniciando evolución del enjambre PSO para 4 hiperparámetros")
        print(f"Parámetros a optimizar: C, gamma, epsilon, tol")
        print(f"Partículas: {self.num_particles} | Iteraciones: {self.max_iter}")
        print("=" * 70)
        
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
            
            # Guardar historial de convergencia
            self.fitness_history.append(self.gbest_fitness)
            
            # Mostrar progreso con formato mejorado
            print(f"Iter {iteration+1:2d}/{self.max_iter} | Mejor Accuracy: {self.gbest_fitness:.4f} | "
                  f"C: {self.gbest_position[0]:6.2f} | γ: {self.gbest_position[1]:.6f} | "
                  f"ε: {self.gbest_position[2]:.5f} | tol: {self.gbest_position[3]:.5f}")

        print("=" * 70)
        print("Evolución completada")
        
        # [Finalización]
        return self.gbest_position, self.gbest_fitness

# ==========================================
# Ejecución y prueba del algoritmo - CON 4 PARÁMETROS
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("OPTIMIZACIÓN DE HIPERPARÁMETROS CON PSO (4 PARÁMETROS)")
    print("=" * 70)
    
    # Cargamos los datos
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    print(f"\n📊 Información del Dataset:")
    print(f"   - Nombre: Breast Cancer Wisconsin")
    print(f"   - Muestras (filas): {X.shape[0]}")
    print(f"   - Características (columnas): {X.shape[1]}")
    print(f"   - Clases: 2 (Maligno=0, Benigno=1)")
    
    # Definimos el espacio de búsqueda (Límites) - 4 PARÁMETROS
    bounds = [
        [0.1, 100.0],      # Límites para C (regularización) - Dim 1: C (Penalización) -> de 0.1 a 100.0
        [0.0001, 2.0],     # Límites para gamma (kernel RBF) - Dim 2: gamma (Coeficiente Kernel) -> de 0.0001 a 1.0
        [0.01, 1.0],       # Límites para epsilon - Dim 3:(Parámetro de tolerancia en la función de pérdida) -> de 0.01 a 1.0
        [0.00001, 0.1]     # Límites para tol (tolerancia) Dim 4:  (Tolerancia para detener el entrenamiento) -> de 1e-5 a 1e-1
    ]
    
    print(f"\n🔍 Espacio de búsqueda (4 dimensiones):")
    print(f"   1. C (regularización): [{bounds[0][0]}, {bounds[0][1]}]")
    print(f"   2. Gamma (kernel): [{bounds[1][0]}, {bounds[1][1]}]")
    print(f"   3. Epsilon: [{bounds[2][0]}, {bounds[2][1]}]")
    print(f"   4. Tol (tolerancia): [{bounds[3][0]}, {bounds[3][1]}]")
    
    # Baseline: SVM con hiperparámetros por defecto de Sklearn
    print("\n" + "-" * 70)
    print("📌 EVALUANDO BASELINE (SVM con parámetros por defecto)")
    print("-" * 70)
    baseline_model = SVC(kernel='rbf')
    baseline_score = cross_val_score(baseline_model, X, y, cv=3, scoring='accuracy').mean()
    print(f"   Accuracy baseline: {baseline_score:.4f}")
    print(f"   Parámetros por defecto: C=1.0, gamma='scale', tol=1e-3")
    
    # Optimizador PSO
    print("\n" + "-" * 70)
    print("🐝 EJECUTANDO PSO PARA OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("-" * 70)
    
    pso = PSOHyperparameterTuning(
        bounds=bounds, 
        num_particles=15, 
        max_iter=15,
        w=0.7,      # Inercia (balance entre exploración/explotación)
        c1=1.5,     # Componente cognitivo
        c2=1.5,     # Componente social
        random_state=42
    )
    best_params, best_accuracy = pso.fit(X, y)

    # Graficar curva de convergencia
    plt.figure(figsize=(8, 5))
    plt.plot(pso.fitness_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Accuracy (Fitness)', fontsize=12)
    plt.title('Curva de Convergencia del Algoritmo PSO', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.0)
    for i, (iter_num, fitness) in enumerate(zip(range(1, len(pso.fitness_history)+1), pso.fitness_history)):
        if i == 0 or i == 3 or i == len(pso.fitness_history)-1:  # Marcar iteraciones clave
            plt.annotate(f'{fitness:.3f}', (iter_num-1, fitness), textcoords="offset points", xytext=(5,5), fontsize=9)
    plt.show()
    
    # Resultados finales
    print("\n" + "=" * 70)
    print("🎯 RESULTADOS FINALES")
    print("=" * 70)
    print(f"\n📈 Comparación de Accuracy:")
    print(f"   Baseline (default):     {baseline_score:.4f}")
    print(f"   PSO optimizado:         {best_accuracy:.4f}")
    print(f"   📊 Mejora:               {((best_accuracy - baseline_score)/baseline_score)*100:.2f}%")
    
    print(f"\n🔧 Mejores Hiperparámetros encontrados:")
    print(f"   • C (regularización):       {best_params[0]:.4f}")
    print(f"   • Gamma (kernel RBF):       {best_params[1]:.6f}")
    print(f"   • Epsilon:                  {best_params[2]:.6f}")
    print(f"   • Tol (tolerancia):         {best_params[3]:.6f}")
    
    # Validación final con los parámetros encontrados
    print(f"\n✅ Validación final del modelo optimizado:")
    best_model = SVC(C=best_params[0], gamma=best_params[1], kernel='rbf', 
                    tol=best_params[3], max_iter=1000)
    final_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f"   Accuracy promedio (5-fold CV): {final_scores.mean():.4f} ± {final_scores.std():.4f}")  
