import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
import warnings

# Suprimir warnings de Sklearn para una salida limpia en consola
warnings.filterwarnings('ignore')

# OPTIMIZADOR ABC
class ABCFeatureSelector:
    """
    Selección de características mediante Artificial Bee Colony (ABC).
    Agnóstico al modelo: Evalúa subconjuntos usando cualquier estimador de Sklearn.
    """
    def __init__(self, estimator, cv, scoring='accuracy', 
                 num_bees=20, max_iter=30, limit=8, 
                 penalty=0.01, guide_prob=0.3, random_state=42):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.num_foods = num_bees // 2
        self.max_iter = max_iter
        self.limit = limit
        self.penalty = penalty
        self.guide_prob = guide_prob
        self.rng = np.random.RandomState(random_state)
        
        self.cache = {}
        self.best_mask = None
        self.best_fitness = -np.inf
        self.global_best = None

    def _binarize(self, continuous_vector):
        return (continuous_vector >= 0.5).astype(np.int8)

    def _evaluate_fitness(self, position, X, y):
        mask = self._binarize(position)
        key = tuple(mask)
        
        # 1. Caché para evitar reentrenar modelos con el mismo subconjunto
        if key in self.cache: 
            return self.cache[key]
        
        n_selected = np.sum(mask)
        if n_selected == 0:
            fitness = -1e6
            self.cache[key] = fitness
            return fitness

        # 2. Evaluación mediante Cross Validation
        X_sel = X[:, mask.astype(bool)]
        model = clone(self.estimator)
        scores = cross_val_score(model, X_sel, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        base_score = scores.mean()

        # 3. Fitness penalizado (promueve menor cantidad de variables)
        fitness = base_score - (self.penalty * (n_selected / X.shape[1]))
        self.cache[key] = fitness
        return fitness

    def fit(self, X, y, verbose=False):
        n_features = X.shape[1]
        
        # Inicialización
        foods = self.rng.uniform(0.0, 1.0, size=(self.num_foods, n_features)).astype(np.float32)
        fitness = np.zeros(self.num_foods, dtype=np.float32)
        trials = np.zeros(self.num_foods, dtype=np.int32)

        for i in range(self.num_foods):
            fitness[i] = self._evaluate_fitness(foods[i], X, y)
            if fitness[i] > self.best_fitness:
                self.best_fitness = fitness[i]
                self.global_best = foods[i].copy()

        # Optimización
        for iteration in range(self.max_iter):
            # Fase Empleadas y Observadoras
            for _ in range(2): 
                for i in range(self.num_foods):
                    k = self.rng.choice([x for x in range(self.num_foods) if x != i])
                    phi = self.rng.uniform(-1, 1, n_features)
                    v = foods[i] + phi * (foods[i] - foods[k])
                    
                    # Guiado hacia el mejor global (GABC)
                    if self.rng.rand() < self.guide_prob and self.global_best is not None:
                        psi = self.rng.uniform(0, 0.5, n_features)
                        v += psi * (self.global_best - foods[i])
                    
                    new_food = np.clip(v, 0.0, 1.0)
                    new_fit = self._evaluate_fitness(new_food, X, y)

                    if new_fit > fitness[i]:
                        foods[i] = new_food
                        fitness[i] = new_fit
                        trials[i] = 0
                    else:
                        trials[i] += 1

            # Fase Exploradoras
            for i in range(self.num_foods):
                if trials[i] > self.limit:
                    foods[i] = self.rng.uniform(0.0, 1.0, n_features).astype(np.float32)
                    fitness[i] = self._evaluate_fitness(foods[i], X, y)
                    trials[i] = 0

            # Actualizar mejor global
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.global_best = foods[best_idx].copy()
                self.best_mask = self._binarize(self.global_best)

            if verbose:
                print(f"Iteración {iteration+1:02d}/{self.max_iter} | "
                      f"Mejor Fitness: {self.best_fitness:.4f} | "
                      f"Variables: {np.sum(self.best_mask)}/{n_features}")

        return self.best_mask

# PREPARACION DE LOS DATOS
def cargar_adult_dataset(ruta_archivo="adult.csv"):
    """Carga, limpia y codifica el dataset."""
    print(f"Cargando datos desde '{ruta_archivo}'...")
    df = pd.read_csv(ruta_archivo, na_values="?")
    df.dropna(inplace=True)
    
    # Target: 1 si es >50K, 0 en caso contrario
    y = df["income"].str.contains(">50K").astype(np.int8).values
    X_df = df.drop(columns=["income"])

    # Codificar variables categóricas a numéricas
    for col in X_df.select_dtypes(include=['object']).columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col])

    return X_df.values.astype(np.float32), y, list(X_df.columns)

# MAIN PRINCIPAL
if __name__ == "__main__":
    # 1. Cargar datos
    try:
        X, y, feature_names = cargar_adult_dataset("../data/adult.csv")
    except FileNotFoundError:
        print("Error: Asegúrate de tener el archivo 'adult.csv' en el mismo directorio.")
        exit(1)
        
    print(f"Dataset cargado con {X.shape[0]} registros y {X.shape[1]} características originales.\n")

    # 2. Definir el modelo base y validación
    modelo_base = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Configurar el algoritmo de enjambre (ABC)
    print("Iniciando optimización por enjambre (ABC)...")
    selector_abc = ABCFeatureSelector(
        estimator=modelo_base,
        cv=cv_strategy,
        scoring='accuracy',
        num_bees=20,       # Tamaño del enjambre
        max_iter=20,       # Iteraciones máximas
        limit=8,           # Paciencia antes de que la exploradora busque nueva fuente
        penalty=0.01,      # Factor de castigo por usar muchas variables
        random_state=42
    )

    # 4. Ejecutar optimización
    mejor_mascara = selector_abc.fit(X, y, verbose=True)

    # 5. Procesar e imprimir resultados
    seleccionadas = [feature_names[i] for i, val in enumerate(mejor_mascara) if val == 1]
    reduccion = 100 * (1 - len(seleccionadas) / len(feature_names))

    print("\n" + "="*40)
    print("          RESULTADOS FINALES")
    print("="*40)
    print(f"Total variables originales: {len(feature_names)}")
    print(f"Total variables seleccionadas: {len(seleccionadas)}")
    print(f"Porcentaje de reducción: {reduccion:.2f}%")
    print("-" * 40)
    print("Variables escogidas por el enjambre:")
    for feature in seleccionadas:
        print(f" - {feature}")
    print("="*40)