"use client";

import { Brain, TrendingUp, Clock, Layers, Target, Settings, Dna, ChevronLeft, ChevronRight, Cpu } from "lucide-react";
import { AGResult }  from "../types";

interface DashboardProps {
  isRunning: boolean;
  result: AGResult | null;
  error: string | null;
  historyIndex: number;
  setHistoryIndex: (i: number) => void;
}

interface StatProps {
  icon: React.ComponentType<{ size?: number }>;
  label: string;
  value: number | string | undefined;
}

function Card({ title, icon: Icon, children }: { title: string; icon: React.ComponentType<{ size?: number }>; children: React.ReactNode }) {
  return (
    <div className="bg-zinc-900 rounded-xl border border-white/5 p-6">
      <div className="flex items-center gap-2 mb-6">
        <Icon size={16} />
        <h2 className="text-sm uppercase font-semibold">{title}</h2>
      </div>
      {children}
    </div>
  );
}

function Stat({ icon: Icon, label, value }: StatProps) {
  return (
    <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
      <div className="flex items-center gap-2 text-zinc-500 text-xs mb-2">
        <Icon size={14} />
        {label}
      </div>
      <div className="text-2xl font-bold text-cyan-400">
        {typeof value === "number" ? value.toFixed(4) : value ?? "—"}
      </div>
    </div>
  );
}

function NavigationControls({ historyIndex, setHistoryIndex, maxIndex }: { historyIndex: number; setHistoryIndex: (i: number) => void; maxIndex: number }) {
  const historyLength = maxIndex + 1;
  const currentStep = historyIndex + 1;

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <button onClick={() => setHistoryIndex(-1)} className={`px-3 py-1.5 text-xs rounded-lg border ${historyIndex === -1 ? "border-cyan-400 text-cyan-400 bg-cyan-400/10" : "border-white/10 text-zinc-500 hover:border-zinc-600"}`}>
          Final
        </button>
        <button onClick={() => setHistoryIndex(Math.max(0, historyIndex - 1))} disabled={historyIndex <= 0} className="p-1.5 rounded-lg border border-white/10 text-zinc-500 disabled:opacity-30">
          <ChevronLeft size={16} />
        </button>
        <span className="text-xs text-zinc-500 px-2">{historyLength > 0 ? `${currentStep} / ${historyLength}` : "—"}</span>
        <button onClick={() => setHistoryIndex(Math.min(maxIndex, historyIndex + 1))} disabled={historyIndex >= maxIndex} className="p-1.5 rounded-lg border border-white/10 text-zinc-500 disabled:opacity-30">
          <ChevronRight size={16} />
        </button>
      </div>
      <div className="text-xs text-zinc-600">{historyIndex === -1 ? "Final result" : `Step ${currentStep}`}</div>
    </div>
  );
}

function StepDetail({ result, stepIndex }: { result: AGResult; stepIndex: number }) {
  const step = result.history?.[stepIndex];
  const isFeatureSelection = !!result.feature_indices;
  const isHyperparameter = !!result.best_hyperparameters;
  const isNeuroevolution = !!result.best_architecture;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <Stat icon={TrendingUp} label="Best Fitness" value={step?.best_fitness} />
        <Stat icon={TrendingUp} label="Best Overall" value={step?.best_overall_fitness} />
      </div>

      {isFeatureSelection && (
        <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
          <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
            <Target size={14} />
            Features Selected
          </div>
          <div className="text-lg font-bold text-cyan-400">{step?.features_selected}</div>
        </div>
      )}

      {isHyperparameter && (
        <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
          <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
            <Settings size={14} />
            Best Hyperparameters
          </div>
          <pre className="text-xs text-zinc-400 overflow-x-auto">{JSON.stringify(result.best_hyperparameters, null, 2)}</pre>
        </div>
      )}

      {isNeuroevolution && (
        <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
          <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
            <Layers size={14} />
            Architecture
          </div>
          <div className="text-lg font-bold text-cyan-400">[{step?.architecture?.join(", ")}]</div>
        </div>
      )}

      <div className="text-xs text-zinc-600 border-t border-white/5 pt-3">
        Generation {step?.generation} • Generation-level snapshot
      </div>
    </div>
  );
}

function FinalResult({ result }: { result: AGResult }) {
  const isFeatureSelection = !!result.feature_indices;
  const isHyperparameter = !!result.best_hyperparameters;
  const isNeuroevolution = !!result.best_architecture;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <Stat icon={TrendingUp} label="Best Fitness" value={result.best_fitness || result.best_accuracy} />
        <Stat icon={Clock} label="Generations" value={result.history?.length} />
      </div>

      {isFeatureSelection && (
        <>
          <div className="grid grid-cols-2 gap-4">
            <Stat icon={Layers} label="Total Features" value={result.total_features} />
            <Stat icon={Target} label="Selected" value={result.selected_features} />
          </div>
          <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
            <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
              <Dna size={14} />
              Selected Features
            </div>
            <div className="flex flex-wrap gap-2">
              {result.feature_names?.map((name: string, i: number) => (
                <span key={i} className="px-2 py-1 text-xs bg-cyan-400/10 text-cyan-400 rounded">
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
          <pre className="text-xs text-zinc-400 overflow-x-auto">{JSON.stringify(result.best_hyperparameters, null, 2)}</pre>
        </div>
      )}

      {isNeuroevolution && (
        <>
          <div className="grid grid-cols-2 gap-4">
            <Stat icon={Layers} label="Total Layers" value={result.total_layers} />
            <Stat icon={Cpu} label="Total Neurons" value={result.total_neurons} />
          </div>
          <div className="bg-zinc-800/50 p-4 rounded-xl border border-white/5">
            <div className="flex items-center gap-2 text-zinc-500 text-xs mb-3">
              <Layers size={14} />
              Best Architecture
            </div>
            <div className="text-lg font-bold text-cyan-400">[{result.best_architecture?.join(" → ")}]</div>
          </div>
        </>
      )}
    </div>
  );
}

export default function Dashboard({ isRunning, result, error, historyIndex, setHistoryIndex }: DashboardProps) {
  const history = result?.history || [];
  const maxIndex = history.length - 1;

  return (
    <section className="flex-1 flex flex-col gap-6">
      <Card title="Execution" icon={Brain}>
        {isRunning && <div className="text-center py-10 text-cyan-400 animate-pulse">Running algorithm...</div>}
        {error && <div className="text-red-400 text-center">{error}</div>}
        {result && (
          <div className="flex flex-col gap-6">
            <NavigationControls historyIndex={historyIndex} setHistoryIndex={setHistoryIndex} maxIndex={maxIndex} />
            {historyIndex >= 0 ? <StepDetail result={result} stepIndex={historyIndex} /> : <FinalResult result={result} />}
          </div>
        )}
      </Card>
    </section>
  );
}