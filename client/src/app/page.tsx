import {
  Brain,
  Shield,
  Activity,
  ArrowRight,
  Cpu,
  Sparkles
} from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col flex-1 items-center justify-center bg-[#131315] text-zinc-200">
      <main className="flex flex-col flex-1 w-full max-w-7xl py-20 px-6">

        {/* HERO */}
        <section className="pt-32 pb-20 grid lg:grid-cols-2 gap-12 items-center">

          <div className="space-y-8">
            <div className="flex items-center gap-2 text-xs uppercase tracking-widest bg-zinc-800 px-3 py-1 rounded-full w-fit">
              <Sparkles className="w-3 h-3 text-cyan-400" />
              v4.0 Neural Architecture Active
            </div>

            <h1 className="text-5xl md:text-7xl font-bold leading-tight">
              The{" "}
              <span className="bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                Neural-Analytical
              </span>{" "}
              Gateway.
            </h1>

            <p className="text-zinc-400 text-lg">
              Harness the power of synaptic computing. Scalable machine learning
              architectures designed for high-resolution neuro-data processing.
            </p>

            <div className="flex gap-4">
              <button className="px-6 py-3 rounded-lg bg-cyan-400 text-black font-semibold">
                Launch Dashboard
              </button>

              <button className="px-6 py-3 rounded-lg border border-zinc-700 hover:bg-zinc-800">
                View Docs
              </button>
            </div>
          </div>

          {/* DASHBOARD PREVIEW */}
          <div className="rounded-xl border border-white/10 bg-zinc-900 shadow-xl">
            <div className="p-4 border-b border-white/10 flex justify-between">
              <div className="flex gap-2">
                <div className="w-3 h-3 bg-red-400 rounded-full" />
                <div className="w-3 h-3 bg-yellow-400 rounded-full" />
                <div className="w-3 h-3 bg-green-400 rounded-full" />
              </div>
              <span className="text-xs text-zinc-500">
                Synaptic-7 // LIVE
              </span>
            </div>

            <div className="p-6 space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-zinc-800 p-4 rounded-lg">
                  <div className="text-xs text-zinc-500">
                    Inference Latency
                  </div>
                  <div className="text-2xl text-cyan-400 font-bold">
                    1.24ms
                  </div>
                </div>

                <div className="bg-zinc-800 p-4 rounded-lg">
                  <div className="text-xs text-zinc-500">
                    Synapse Weight
                  </div>
                  <div className="text-2xl text-purple-400 font-bold">
                    0.982
                  </div>
                </div>
              </div>

              <div className="flex items-end gap-1 h-24">
                {[60, 45, 80, 55, 90, 40, 70].map((h, i) => (
                  <div
                    key={i}
                    style={{ height: `${h}%` }}
                    className="bg-cyan-400/30 w-full rounded-t"
                  />
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* FEATURES */}
        <section className="py-20">
          <h2 className="text-3xl font-bold mb-12">
            Engineered for Precision
          </h2>

          <div className="grid md:grid-cols-3 gap-6">

            <div className="md:col-span-2 p-8 rounded-xl bg-zinc-900 border border-white/5 space-y-4">
              <Brain className="text-cyan-400" size={32} />
              <h3 className="text-xl font-semibold">
                High-performance neuro-analytics
              </h3>
              <p className="text-zinc-400">
                Process petabytes of neural correlation data with real-time
                mapping of brain-machine interface streams.
              </p>
            </div>

            <div className="p-8 rounded-xl bg-zinc-900 border border-white/5 space-y-4">
              <Cpu className="text-purple-400" size={32} />
              <h3 className="text-xl font-semibold">
                Scalable Model Lab
              </h3>
              <p className="text-zinc-400">
                Stress-test neuro models against billions of synthetic patterns.
              </p>
            </div>

            <div className="p-8 rounded-xl bg-zinc-900 border border-white/5 space-y-4">
              <Activity className="text-orange-400" size={32} />
              <h3 className="text-xl font-semibold">
                Real-time EEG simulation
              </h3>
              <p className="text-zinc-400">
                Simulate high-fidelity EEG responses for BCI testing.
              </p>
            </div>

            <div className="md:col-span-2 p-8 rounded-xl bg-gradient-to-br from-zinc-900 to-zinc-800 border border-white/5 space-y-4">
              <Shield className="text-cyan-400" size={32} />
              <h3 className="text-xl font-semibold">
                Global Security Lattice
              </h3>
              <p className="text-zinc-400">
                Enterprise-grade encryption for sensitive neuro-metric data.
              </p>

              <button className="flex items-center gap-2 text-cyan-400 font-semibold">
                Explore Security
                <ArrowRight size={16} />
              </button>
            </div>

          </div>
        </section>

        {/* CTA */}
        <section className="py-24 text-center space-y-6">
          <h2 className="text-4xl font-bold">
            Ready to bridge the{" "}
            <span className="text-cyan-400 italic">
              synaptic gap
            </span>
            ?
          </h2>

          <p className="text-zinc-400 max-w-2xl mx-auto">
            Join world-class neuro-engineering labs using CognitiveHub.
          </p>

          <div className="flex justify-center gap-4">
            <button className="px-6 py-3 bg-white text-black rounded-lg font-semibold">
              Start Building
            </button>
            <button className="px-6 py-3 border border-zinc-700 rounded-lg">
              Contact Sales
            </button>
          </div>
        </section>

        {/* FOOTER */}
        <footer className="border-t border-white/5 pt-8 text-sm text-zinc-500 flex flex-col md:flex-row justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-cyan-400 rounded" />
            CognitiveHub
          </div>

          <div className="flex gap-6">
            <a>Privacy</a>
            <a>Terms</a>
            <a>API</a>
            <a>Github</a>
          </div>

          <div>© 2026 CognitiveHub</div>
        </footer>

      </main>
    </div>
  );
}