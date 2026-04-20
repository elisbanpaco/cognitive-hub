import { useState, useCallback } from "react";
import { AlgorithmType, AGConfig, AGResult } from "./types";
import { getDefaultConfig } from "./config";

const baseUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/v1/AG`;

export function useAG() {
  const [algorithm, setAlgorithm] = useState<AlgorithmType>("feature_selection");
  const [config, setConfig] = useState<AGConfig>(getDefaultConfig("feature_selection"));
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<AGResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const changeAlgorithm = useCallback((newAlgo: AlgorithmType) => {
    setAlgorithm(newAlgo);
    setConfig(getDefaultConfig(newAlgo));
  }, []);

  const runAlgorithm = useCallback(async () => {
    setIsRunning(true);
    setError(null);
    setResult(null);
    setHistoryIndex(-1);

    try {
      let endpoint = "";
      switch (algorithm) {
        case "feature_selection":
          endpoint = "/feature-selection";
          break;
        case "hyperparameter_optimization":
          endpoint = "/hyperparameter-optimization";
          break;
        case "neuroevolution":
          endpoint = "/neuroevolution";
          break;
      }

      const response = await fetch(`${baseUrl}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      if (!response.ok) throw new Error("API Error");

      const data = await response.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsRunning(false);
    }
  }, [algorithm, config]);

  return {
    algorithm,
    changeAlgorithm,
    config,
    setConfig,
    isRunning,
    result,
    error,
    historyIndex,
    setHistoryIndex,
    runAlgorithm,
  };
}
