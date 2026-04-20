"use client";

import { Settings, Play, Cpu, Target, Brain, CheckCircle } from "lucide-react";
import { AlgorithmType, AGConfig } from "../types";

interface ConfigPanelProps {
  algorithm: AlgorithmType;
  setAlgorithm: (a: AlgorithmType) => void;
  config: AGConfig;
  setConfig: (c: AGConfig) => void;
  isRunning: boolean;
  run: () => void;
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}

interface OptionProps {
  icon: React.ComponentType<{ size?: number }>;
  label: string;
  active: boolean;
  onClick: () => void;
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

function Slider({ label, value, min, max, step = 1, disabled, onChange }: SliderProps) {
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
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

function PrimaryButton({ children, icon: Icon, ...props }: { children: React.ReactNode; icon: React.ComponentType<{ size?: number }>; onClick?: () => void }) {
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

function Option({ icon: Icon, label, active, onClick }: OptionProps) {
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

export default function ConfigPanel({ algorithm, setAlgorithm, config, setConfig, isRunning, run }: ConfigPanelProps) {
  return (
    <section className="w-80 flex flex-col gap-6 overflow-y-auto">
      <Card title="Algorithm Configuration" icon={Settings}>
        <div className="space-y-2 mb-5">
          <Option label="Feature Selection" icon={Target} active={algorithm === "feature_selection"} onClick={() => setAlgorithm("feature_selection")} />
          <Option label="Hyperparameter" icon={Cpu} active={algorithm === "hyperparameter_optimization"} onClick={() => setAlgorithm("hyperparameter_optimization")} />
          <Option label="Neuroevolution" icon={Brain} active={algorithm === "neuroevolution"} onClick={() => setAlgorithm("neuroevolution")} />
        </div>

        <Slider label="Population" value={config.population_size} min={5} max={50} disabled={isRunning} onChange={(v) => setConfig({ ...config, population_size: v })} />
        <Slider label="Generations" value={config.generations} min={5} max={50} disabled={isRunning} onChange={(v) => setConfig({ ...config, generations: v })} />
        <Slider label="Mutation" value={config.mutation_rate} min={0.01} max={0.5} step={0.01} disabled={isRunning} onChange={(v) => setConfig({ ...config, mutation_rate: v })} />

        <PrimaryButton icon={Play} onClick={run}>Run Algorithm</PrimaryButton>
      </Card>
    </section>
  );
}