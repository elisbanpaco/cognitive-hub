"use client";

import { Terminal } from "lucide-react";
import { useAG } from "./useAG";
import ConfigPanel from "./_components/ConfigPanel";
import Dashboard from "./_components/Dashboard";

function Header() {
  return (
    <header className="h-16 border-b border-white/5 flex items-center px-6">
      <div className="flex items-center gap-3">
        <Terminal className="text-cyan-400" size={18} />
        <h1 className="font-semibold uppercase text-sm">Model Lab / Genetic Algorithm Engine</h1>
      </div>
    </header>
  );
}

export default function AGPage() {
  const { algorithm, changeAlgorithm, config, setConfig, isRunning, result, error, historyIndex, setHistoryIndex, runAlgorithm } = useAG();

  return (
    <div className="flex bg-[#131315] text-zinc-200 h-screen overflow-hidden">
      <main className="flex-1 flex flex-col">
        <Header />
        <div className="flex flex-1 p-6 gap-6 overflow-hidden">
          <ConfigPanel
            algorithm={algorithm}
            setAlgorithm={changeAlgorithm}
            config={config}
            setConfig={setConfig}
            isRunning={isRunning}
            run={runAlgorithm}
          />
          <Dashboard
            algorithm={algorithm}
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