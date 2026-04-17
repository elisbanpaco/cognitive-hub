import numpy as np
import random
import warnings
from typing import Callable, Dict, Any, List, Optional
from enum import Enum
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class AGType(str, Enum):
    FEATURE_SELECTION = "feature_selection"
    HYPERPARAMETER_OPT = "hyperparameter_optimization"
    NEUROEVOLUTION = "neuroevolution"


class AGEngine:
    def __init__(self, websocket_callback: Optional[Callable] = None):
        self.websocket_callback = websocket_callback
        self.data = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = None
        self.best_overall = None

    async def _send_progress(self, message: Dict[str, Any]):
        if self.websocket_callback:
            await self.websocket_callback(message)

    def set_websocket(self, callback: Callable):
        self.websocket_callback = callback

    def load_data(self, dataset_name: str = "breast_cancer"):
        if dataset_name == "breast_cancer":
            self.data = load_breast_cancer()
            self.X, self.y = self.data.data, self.data.target
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X)
        elif dataset_name == "make_classification":
            self.X, self.y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=42,
            )
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X)

        return {
            "dataset": dataset_name,
            "X_shape": self.X.shape,
            "y_shape": self.y.shape,
        }

    async def run_feature_selection(
        self,
        population_size: int = 20,
        generations: int = 15,
        mutation_rate: float = 0.05,
        tournament_size: int = 3,
    ):
        if self.X is None:
            self.load_data()

        n_features = self.X.shape[1]
        clf = RandomForestClassifier(n_estimators=20, random_state=42)

        def calculate_fitness(chromosome):
            selected_indices = np.where(chromosome == 1)[0]
            if len(selected_indices) == 0:
                return 0.0
            X_subset = self.X[:, selected_indices]
            scores = cross_val_score(clf, X_subset, self.y, cv=3, scoring="accuracy")
            return scores.mean()

        def init_population(pop_size, n_feats):
            return [np.random.randint(0, 2, n_feats) for _ in range(pop_size)]

        def tournament_selection(population, fitnesses):
            selected_indices = random.sample(range(len(population)), tournament_size)
            best_index = max(selected_indices, key=lambda i: fitnesses[i])
            return population[best_index]

        def crossover(parent1, parent2):
            if random.random() < 0.8:
                point = random.randint(1, n_features - 1)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
                return child1, child2
            return parent1.copy(), parent2.copy()

        def mutate(chromosome):
            for i in range(n_features):
                if random.random() < mutation_rate:
                    chromosome[i] = 1 - chromosome[i]
            return chromosome

        population = init_population(population_size, n_features)
        best_overall_fitness = 0.0
        best_overall_chromosome = None
        history = []

        await self._send_progress(
            {
                "type": "start",
                "algorithm": "feature_selection",
                "population_size": population_size,
                "generations": generations,
                "n_features": n_features,
            }
        )

        for generation in range(generations):
            fitnesses = [calculate_fitness(ind) for ind in population]

            best_gen_fitness = max(fitnesses)
            best_gen_idx = fitnesses.index(best_gen_fitness)
            best_gen_chromosome = population[best_gen_idx]

            if best_gen_fitness > best_overall_fitness:
                best_overall_fitness = best_gen_fitness
                best_overall_chromosome = best_gen_chromosome.copy()

            num_features_selected = np.sum(best_gen_chromosome)
            history.append(
                {
                    "generation": generation + 1,
                    "best_fitness": float(best_gen_fitness),
                    "best_overall_fitness": float(best_overall_fitness),
                    "features_selected": int(num_features_selected),
                }
            )

            await self._send_progress(
                {
                    "type": "generation",
                    "generation": generation + 1,
                    "best_fitness": float(best_gen_fitness),
                    "features_selected": int(num_features_selected),
                    "history": history,
                }
            )

            new_population = [best_gen_chromosome]

            while len(new_population) < population_size:
                p1 = tournament_selection(population, fitnesses)
                p2 = tournament_selection(population, fitnesses)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                new_population.extend([c1, c2])

            population = new_population[:population_size]

        final_features = np.where(best_overall_chromosome == 1)[0]
        result = {
            "type": "complete",
            "best_fitness": float(best_overall_fitness),
            "total_features": n_features,
            "selected_features": len(final_features),
            "feature_indices": final_features.tolist(),
            "feature_names": [self.data.feature_names[idx] for idx in final_features],
            "history": history,
        }

        await self._send_progress(result)
        return result

    async def run_hyperparameter_optimization(
        self,
        population_size: int = 10,
        generations: int = 10,
        mutation_rate: float = 0.2,
    ):
        if self.X is None:
            self.load_data()

        search_space = {
            "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
            "activation": ["tanh", "relu", "logistic"],
            "alpha": (0.0001, 0.1),
            "learning_rate_init": (0.001, 0.1),
        }

        def create_individual():
            return {
                "hidden_layer_sizes": random.choice(search_space["hidden_layer_sizes"]),
                "activation": random.choice(search_space["activation"]),
                "alpha": random.uniform(
                    search_space["alpha"][0], search_space["alpha"][1]
                ),
                "learning_rate_init": random.uniform(
                    search_space["learning_rate_init"][0],
                    search_space["learning_rate_init"][1],
                ),
            }

        def calculate_fitness(individual):
            model = MLPClassifier(
                hidden_layer_sizes=individual["hidden_layer_sizes"],
                activation=individual["activation"],
                alpha=individual["alpha"],
                learning_rate_init=individual["learning_rate_init"],
                max_iter=100,
                random_state=42,
            )
            scores = cross_val_score(
                model, self.X_scaled, self.y, cv=3, scoring="accuracy"
            )
            return scores.mean()

        def tournament_selection(population, fitnesses, k=3):
            selected_indices = random.sample(range(len(population)), k)
            best_index = max(selected_indices, key=lambda i: fitnesses[i])
            return population[best_index].copy()

        def crossover(parent1, parent2):
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
            if random.random() < mutation_rate:
                gene_to_mutate = random.choice(list(individual.keys()))
                if gene_to_mutate == "hidden_layer_sizes":
                    individual[gene_to_mutate] = random.choice(
                        search_space["hidden_layer_sizes"]
                    )
                elif gene_to_mutate == "activation":
                    individual[gene_to_mutate] = random.choice(
                        search_space["activation"]
                    )
                elif gene_to_mutate == "alpha":
                    individual[gene_to_mutate] = random.uniform(
                        search_space["alpha"][0], search_space["alpha"][1]
                    )
                elif gene_to_mutate == "learning_rate_init":
                    individual[gene_to_mutate] = random.uniform(
                        search_space["learning_rate_init"][0],
                        search_space["learning_rate_init"][1],
                    )
            return individual

        population = [create_individual() for _ in range(population_size)]
        best_overall_fitness = 0.0
        best_overall_ind = None
        history = []

        await self._send_progress(
            {
                "type": "start",
                "algorithm": "hyperparameter_optimization",
                "population_size": population_size,
                "generations": generations,
                "search_space": {
                    k: str(v) if not isinstance(v, tuple) else v
                    for k, v in search_space.items()
                },
            }
        )

        for generation in range(generations):
            fitnesses = [calculate_fitness(ind) for ind in population]

            best_gen_fitness = max(fitnesses)
            best_gen_idx = fitnesses.index(best_gen_fitness)
            best_gen_ind = population[best_gen_idx]

            if best_gen_fitness > best_overall_fitness:
                best_overall_fitness = best_gen_fitness
                best_overall_ind = best_gen_ind.copy()

            history.append(
                {
                    "generation": generation + 1,
                    "best_fitness": float(best_gen_fitness),
                    "best_overall_fitness": float(best_overall_fitness),
                }
            )

            await self._send_progress(
                {
                    "type": "generation",
                    "generation": generation + 1,
                    "best_fitness": float(best_gen_fitness),
                    "best_individual": {
                        k: str(v) if isinstance(v, tuple) else v
                        for k, v in best_gen_ind.items()
                    },
                    "history": history,
                }
            )

            new_population = [best_gen_ind]

            while len(new_population) < population_size:
                p1 = tournament_selection(population, fitnesses)
                p2 = tournament_selection(population, fitnesses)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                new_population.extend([c1, c2])

            population = new_population[:population_size]

        result = {
            "type": "complete",
            "best_fitness": float(best_overall_fitness),
            "best_hyperparameters": {
                k: str(v) if isinstance(v, tuple) else v
                for k, v in best_overall_ind.items()
            },
            "history": history,
        }

        await self._send_progress(result)
        return result

    async def run_neuroevolution(
        self,
        population_size: int = 15,
        generations: int = 12,
        mutation_rate: float = 0.3,
        max_layers: int = 5,
        max_neurons: int = 128,
    ):
        if self.X is None:
            self.load_data("make_classification")

        def create_chromosome():
            chromosome = []
            for _ in range(max_layers):
                if random.random() < 0.3:
                    chromosome.append(0)
                else:
                    chromosome.append(random.randint(4, max_neurons))
            return chromosome

        def decode_chromosome(chromosome):
            architecture = tuple([neurons for neurons in chromosome if neurons > 0])
            return architecture if len(architecture) > 0 else (4,)

        def calculate_fitness(chromosome):
            architecture = decode_chromosome(chromosome)
            model = MLPClassifier(
                hidden_layer_sizes=architecture,
                activation="relu",
                max_iter=150,
                random_state=42,
            )
            accuracy = cross_val_score(
                model, self.X_scaled, self.y, cv=3, scoring="accuracy"
            ).mean()
            total_neurons = sum(architecture)
            total_layers = len(architecture)
            fitness = accuracy - (0.0001 * total_neurons) - (0.001 * total_layers)
            return fitness, accuracy, architecture

        population = [create_chromosome() for _ in range(population_size)]
        best_overall_fitness = -float("inf")
        best_overall_acc = 0
        best_architecture = None
        best_chromosome = None
        history = []

        await self._send_progress(
            {
                "type": "start",
                "algorithm": "neuroevolution",
                "population_size": population_size,
                "generations": generations,
                "max_layers": max_layers,
                "max_neurons": max_neurons,
            }
        )

        for generation in range(generations):
            evaluated = [calculate_fitness(ind) for ind in population]
            fitnesses = [eval[0] for eval in evaluated]
            accuracies = [eval[1] for eval in evaluated]
            architectures = [eval[2] for eval in evaluated]

            best_gen_idx = np.argmax(fitnesses)

            if fitnesses[best_gen_idx] > best_overall_fitness:
                best_overall_fitness = fitnesses[best_gen_idx]
                best_overall_acc = accuracies[best_gen_idx]
                best_architecture = architectures[best_gen_idx]
                best_chromosome = population[best_gen_idx].copy()

            history.append(
                {
                    "generation": generation + 1,
                    "best_accuracy": float(accuracies[best_gen_idx]),
                    "best_overall_accuracy": float(best_overall_acc),
                    "architecture": architectures[best_gen_idx],
                }
            )

            await self._send_progress(
                {
                    "type": "generation",
                    "generation": generation + 1,
                    "best_accuracy": float(accuracies[best_gen_idx]),
                    "best_overall_accuracy": float(best_overall_acc),
                    "architecture": architectures[best_gen_idx],
                    "history": history,
                }
            )

            new_population = [best_chromosome]

            while len(new_population) < population_size:
                t1 = random.sample(range(population_size), 3)
                p1 = population[max(t1, key=lambda i: fitnesses[i])]
                t2 = random.sample(range(population_size), 3)
                p2 = population[max(t2, key=lambda i: fitnesses[i])]
                c1, c2 = crossover(p1, p2)
                new_population.extend([mutate(c1), mutate(c2)])

            population = new_population[:population_size]

        result = {
            "type": "complete",
            "best_accuracy": float(best_overall_acc),
            "best_architecture": best_architecture,
            "total_layers": len(best_architecture),
            "total_neurons": sum(best_architecture),
            "history": history,
        }

        await self._send_progress(result)
        return result


async def run_algorithm(
    algorithm_type: AGType, websocket_callback: Optional[Callable] = None, **kwargs
):
    engine = AGEngine(websocket_callback=websocket_callback)

    if algorithm_type == AGType.FEATURE_SELECTION:
        return await engine.run_feature_selection(**kwargs)
    elif algorithm_type == AGType.HYPERPARAMETER_OPT:
        return await engine.run_hyperparameter_optimization(**kwargs)
    elif algorithm_type == AGType.NEUROEVOLUTION:
        return await engine.run_neuroevolution(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome, mutation_rate=0.3, max_neurons=128):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            if chromosome[i] > 0 and random.random() < 0.2:
                chromosome[i] = 0
            else:
                chromosome[i] = random.randint(4, max_neurons)
    return chromosome
