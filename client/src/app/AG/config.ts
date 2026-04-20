import { AlgorithmType, AGConfig } from "./types";

export const getDefaultConfig = (algo: AlgorithmType): AGConfig => {
  switch (algo) {
    case "feature_selection":
      return {
        dataset: "breast_cancer",
        population_size: 20,
        generations: 15,
        mutation_rate: 0.05,
        tournament_size: 3,
      };

    case "hyperparameter_optimization":
      return {
        dataset: "breast_cancer",
        population_size: 10,
        generations: 10,
        mutation_rate: 0.2,
        tournament_size: 3,
      };

    case "neuroevolution":
      return {
        dataset: "breast_cancer",
        population_size: 15,
        generations: 12,
        mutation_rate: 0.3,
        tournament_size: 3,
        max_layers: 5,
        max_neurons: 128,
      };
  }
};
