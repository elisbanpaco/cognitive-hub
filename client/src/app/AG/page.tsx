
"use client";

import { useState } from "react";
import {
  Terminal,
  Sliders,
  Play,
  Brain,
  CheckCircle,
  Cpu,
  Target,
  TrendingUp,
  Clock,
  ChevronLeft,
  ChevronRight,
  Layers,
  Dna,
  Settings,
} from "lucide-react";

type AlgorithmType =
  | "feature_selection"
  | "hyperparameter_optimization"
  | "neuroevolution";

interface AGConfig {
  dataset: string;
  population_size: number;
  generations: number;
  mutation_rate: number;
  tournament_size: number;
  max_layers?: number;
  max_neurons?: number;
}

const getDefaultConfig = (algo: AlgorithmType): AGConfig => {
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

export default function AG() {
  const baseUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/v1/AG`;

  const [algorithm, setAlgorithm] =
    useState<AlgorithmType>("feature_selection");

  const [config, setConfig] = useState<AGConfig>(
    getDefaultConfig("feature_selection")
  );

  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const runAlgorithm = async () => {
    setIsRunning(true);
    setError(null);
    setResult(null);
    setHistoryIndex(-1);

    try {
      let endpoint = "";

      if (algorithm === "feature_selection")
        endpoint = "/feature-selection";

      if (algorithm === "hyperparameter_optimization")
        endpoint = "/hyperparameter-optimization";

      if (algorithm === "neuroevolution")
        endpoint = "/neuroevolution";

      const response = await fetch(`${baseUrl}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) throw new Error("API Error");

      const data = await response.json();

      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="flex bg-[#131315] text-zinc-200 h-screen overflow-hidden">
      <main className="flex-1 flex flex-col">
        <Header />

        <div className="flex flex-1 p-6 gap-6 overflow-hidden">
          <ConfigPanel
            algorithm={algorithm}
            setAlgorithm={(a: AlgorithmType) => {
              setAlgorithm(a);
              setConfig(getDefaultConfig(a));
            }}
            config={config}
            setConfig={setConfig}
            isRunning={isRunning}
            run={runAlgorithm}
          />

          <Dashboard
            isRunning={isRunning}
            result={result}
            error={error}
            historyIndex={historyIndex}
            setHistoryIndex={setHistoryIndex}
          />
        </div>
      </main>
    </div>
  );
}

function Header() {
  return (
    <header className="h-16 border-b border-white/5 flex items-center px-6">
      <div className="flex items-center gap-3">
        <Terminal className="text-cyan-400" size={18} />
        <h1 className="font-semibold uppercase text-sm">
          Model Lab / Genetic Algorithm Engine
        </h1>
      </div>
    </header>
  );
}

function ConfigPanel({
  algorithm,
  setAlgorithm,
  config,
  setConfig,
  isRunning,
  run,
}: any) {
  return (
    <section className="w-80 flex flex-col gap-6 overflow-y-auto">
      <Card title="Algorithm Configuration" icon={Sliders}>
        <AlgorithmSelector
          algorithm={algorithm}
          setAlgorithm={setAlgorithm}
        />

        <Slider
          label="Population"
          value={config.population_size}
          min={5}
          max={50}
          disabled={isRunning}
          onChange={(v: number) =>
            setConfig({ ...config, population_size: v })
          }
        />

        <Slider
          label="Generations"
          value={config.generations}
          min={5}
          max={50}
          disabled={isRunning}
          onChange={(v: number) =>
            setConfig({ ...config, generations: v })
          }
        />

        <Slider
          label="Mutation"
          value={config.mutation_rate}
          min={0.01}
          max={0.5}
          step={0.01}
          disabled={isRunning}
          onChange={(v: number) =>
            setConfig({ ...config, mutation_rate: v })
          }
        />

        <PrimaryButton icon={Play} onClick={run}>
          Run Algorithm
        </PrimaryButton>
      </Card>
    </section>
  );
}

function Dashboard({ isRunning, result, error, historyIndex, setHistoryIndex }: any) {
  const history = result?.history || [];
  const maxIndex = history.length - 1;

  return (
    <section className="flex-1 flex flex-col gap-6">
      <Card title="Execution" icon={Brain}>
        {isRunning && (
          <div className="text-center py-10 text-cyan-400 animate-pulse">
            Running algorithm...
          </div>
        )}

        {error && <div className="text-red-400 text-center">{error}</div>}

        {result && (
          <div className="flex flex-col gap-6">
            <NavigationControls
              historyIndex={historyIndex}
              setHistoryIndex={setHistoryIndex}
              maxIndex={maxIndex}
            />

            {historyIndex >= 0 ? (
              <StepDetail result={result} stepIndex={historyIndex} />
            ) : (
              <FinalResult result={result} />
            )}
          </div>
        )}
      </Card>
    </section>
  );
}

function NavigationControls({
  historyIndex,
  setHistoryIndex,
  maxIndex,
}: any) {
  const historyLength = maxIndex + 1;
  const currentStep = historyIndex + 1;

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <button
          onClick={() => setHistoryIndex(-1)}
          className={`px-3 py-1.5 text-xs rounded-lg border ${
            historyIndex === -1
              ? "border-cyan-400 text-cyan-400 bg-cyan-400/10"
              : "border-white/10 text-zinc-500 hover:border-zinc-600"
          }`}
        >
          Final
        </button>

        <button
          onClick={() => setHistoryIndex(Math.max(0, historyIndex - 1))}
          disabled={historyIndex <= 0}
          className="p-1.5 rounded-lg border border-white/10 text-zinc-500 disabled:opacity-30"
        >
          <ChevronLeft size={16} />
        </button>

        <span className="text-xs text-zinc-500 px-2">
          {historyLength > 0
            ? `${currentStep} / ${historyLength}`
            : "—"}
        </span>

        <button
          onClick={() => setHistoryIndex(Math.min(maxIndex, historyIndex + 1))}
          disabled={historyIndex >= maxIndex}
          className="p-1.5 rounded-lg border border-white/10 text-zinc-500 disabled:opacity-30"
        >
          <ChevronRight size={16} />
        </button>
      </div>

      <div className="text-xs text-zinc-600">
        {historyIndex === -1 ? "Final result" : `Step ${currentStep}`}
      </div>
    </div>
  );
}

function StepDetail({ result, stepIndex }: any) {
  const step = result.history[stepIndex];
  const isFeatureSelection = result.type === "complete" && "feature_indices" in result;
  const isHyperparameter = result.type === "complete" && "best_hyperparameters" in result;
  const isNeuroevolution = result.type === "complete" && "best_architecture" in result;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <Stat
          icon={TrendingUp}
          label="Best Fitness"
          value={step.best_fitness}
        />
        <Stat
          icon={TrendingUp}
          label="Best Overall"
          value={step.best_overall_fitness}
        />
      </div>

      {isFeatureSelection && (
        <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
          <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
            <Target size={14} />
            Features Selected
          </div>
          <div className="text-lg font-bold text-cyan-400">
            {step.features_selected}
          </div>
        </div>
      )}

      {isHyperparameter && (
        <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
          <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
            <Settings size={14} />
            Best Hyperparameters
          </div>
          <pre className="text-xs text-zinc-400 overflow-x-auto">
            {JSON.stringify(result.best_hyperparameters, null, 2)}
          </pre>
        </div>
      )}

      {isNeuroevolution && (
        <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
          <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
            <Layers size={14} />
            Architecture
          </div>
          <div className="text-lg font-bold text-cyan-400">
            [{step.architecture?.join(", ")}]
          </div>
        </div>
      )}

      <div className="text-xs text-zinc-600 border-t border-white/5 pt-3">
        Generation {step.generation} • Generation-level snapshot
      </div>
    </div>
  );
}

function FinalResult({ result }: any) {
  const isFeatureSelection = result.type === "complete" && "feature_indices" in result;
  const isHyperparameter = result.type === "complete" && "best_hyperparameters" in result;
  const isNeuroevolution = result.type === "complete" && "best_architecture" in result;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <Stat
          icon={TrendingUp}
          label="Best Fitness"
          value={result.best_fitness || result.best_accuracy}
        />
        <Stat
          icon={Clock}
          label="Generations"
          value={result.history?.length}
        />
      </div>

      {isFeatureSelection && (
        <>
          <div className="grid grid-cols-2 gap-4">
            <Stat
              icon={Layers}
              label="Total Features"
              value={result.total_features}
            />
            <Stat
              icon={Target}
              label="Selected"
              value={result.selected_features}
            />
          </div>

          <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
            <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
              <Dna size={14} />
              Selected Features
            </div>
            <div className="flex flex-wrap gap-2">
              {result.feature_names?.map((name: string, i: number) => (
                <span
                  key={i}
                  className="px-2 py-1 text-xs bg-cyan-400/10 text-cyan-400 rounded"
                >
                  {name}
                </span>
              ))}
            </div>
          </div>
        </>
      )}

      {isHyperparameter && (
        <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
          <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
            <Settings size={14} />
            Best Hyperparameters
          </div>
          <pre className="text-xs text-zinc-400 overflow-x-auto">
            {JSON.stringify(result.best_hyperparameters, null, 2)}
          </pre>
        </div>
      )}

      {isNeuroevolution && (
        <>
          <div className="grid grid-cols-2 gap-4">
            <Stat
              icon={Layers}
              label="Total Layers"
              value={result.total_layers}
            />
            <Stat
              icon={Cpu}
              label="Total Neurons"
              value={result.total_neurons}
            />
          </div>

          <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
            <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
              <Layers size={14} />
              Best Architecture
            </div>
            <div className="text-lg font-bold text-cyan-400">
              [{result.best_architecture?.join(" → ")}]
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Card({ title, icon: Icon, children }: any) {
  return (
    <div className="bg-zinc-900 rounded-xl border border-white/5 p-6">
      <div className="flex items-center gap-2 mb-6">
        <Icon size={16} />
        <h2 className="text-sm uppercase font-semibold">
          {title}
        </h2>
      </div>
      {children}
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  disabled,
}: any) {
  return (
    <div className="mb-5">
      <div className="flex justify-between text-xs mb-2">
        <span>{label}</span>
        <span className="text-cyan-400">{value}</span>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) =>
          onChange(parseFloat(e.target.value))
        }
        className="w-full"
      />
    </div>
  );
}

function PrimaryButton({ children, icon: Icon, ...props }: any) {
  return (
    <button
      {...props}
      className="w-full py-3 bg-gradient-to-br from-cyan-400 to-cyan-600 text-black rounded-lg font-semibold flex items-center justify-center gap-2"
    >
      <Icon size={16} />
      {children}
    </button>
  );
}

function AlgorithmSelector({ algorithm, setAlgorithm }: any) {
  return (
    <div className="space-y-2 mb-5">
      <Option
        label="Feature Selection"
        icon={Target}
        active={algorithm === "feature_selection"}
        onClick={() => setAlgorithm("feature_selection")}
      />

      <Option
        label="Hyperparameter"
        icon={Cpu}
        active={algorithm === "hyperparameter_optimization"}
        onClick={() =>
          setAlgorithm("hyperparameter_optimization")
        }
      />

      <Option
        label="Neuroevolution"
        icon={Brain}
        active={algorithm === "neuroevolution"}
        onClick={() => setAlgorithm("neuroevolution")}
      />
    </div>
  );
}

function Stat({ icon: Icon, label, value }: any) {
  return (
    <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
      <div className="flex items-center gap-2 text-zinc-500 text-xs mb-2">
        <Icon size={14} />
        {label}
      </div>

      <div className="text-2xl font-bold text-cyan-400">
        {typeof value === "number"
          ? value.toFixed(4)
          : value}
      </div>
    </div>
  );
}

function Option({
  icon: Icon,
  label,
  active,
  onClick,
}: any) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center justify-between p-3 rounded-lg border ${
        active
          ? "border-cyan-400 text-cyan-400 bg-cyan-400/10"
          : "border-white/5 text-zinc-500 hover:border-zinc-700"
      }`}
    >
      <div className="flex items-center gap-2">
        <Icon size={16} />
        {label}
      </div>
      {active && <CheckCircle size={14} />}
    </button>
  );
}