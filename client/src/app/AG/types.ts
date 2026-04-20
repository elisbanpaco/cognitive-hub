export type AlgorithmType =
  | "feature_selection"
  | "hyperparameter_optimization"
  | "neuroevolution";

export interface AGConfig {
  dataset: string;
  population_size: number;
  generations: number;
  mutation_rate: number;
  tournament_size: number;
  max_layers?: number;
  max_neurons?: number;
}

export interface AGResult {
  type: "complete";
  best_fitness?: number;
  best_accuracy?: number;
  history?: AGStep[];
  feature_indices?: number[];
  feature_names?: string[];
  total_features?: number;
  selected_features?: number;
  best_hyperparameters?: Record<string, number | string | boolean>;
  best_architecture?: number[];
  total_layers?: number;
  total_neurons?: number;
}

export interface AGStep {
  generation: number;
  best_fitness: number;
  best_overall_fitness: number;
  features_selected?: number;
  architecture?: number[];
}
