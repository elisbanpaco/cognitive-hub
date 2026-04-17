import Link from "next/link";
export default function Navbar() {
    return (
  
        <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-[#1c1b1d]/70 border-b border-white/5">
          <div className="max-w-7xl mx-auto flex items-center justify-between h-16 px-6">
            <div className="font-bold tracking-widest text-cyan-400">
              <Link href="/">
                CognitiveHub
              </Link>
              
            </div>

            <div className="hidden md:flex gap-8 text-sm">
              <Link className="text-cyan-400 border-b border-cyan-400 pb-1" href="/">
                Neural Engine
              </Link>
              <a className="text-zinc-400 hover:text-white">Analytics</a>
              <a className="text-zinc-400 hover:text-white" href="/AG">Algoritmos Geneticos</a>
              <a className="text-zinc-400 hover:text-white">Architecture</a>
              <a className="text-zinc-400 hover:text-white">Security</a>
            </div>

            <div className="flex gap-3">
              <button className="text-zinc-400 hover:text-cyan-300">
                Login
              </button>
              <button className="px-4 py-2 rounded-lg bg-gradient-to-br from-cyan-400 to-cyan-600 text-black font-semibold">
                Get Started
              </button>
            </div>
          </div>
        </nav>
    );
}