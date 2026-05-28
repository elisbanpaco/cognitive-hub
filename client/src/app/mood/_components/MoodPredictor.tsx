'use client';

import { useState } from 'react';

interface PredictionResponse {
  status: string;
  predicted_mood: string;
}

export default function MoodPredictor() {
  const [features, setFeatures] = useState({
    danceability: 0.5,
    energy: 0.5,
    acousticness: 0.5,
    tempo: 120,
  });
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSliderChange = (key: string, value: number) => {
    setFeatures((prev) => ({ ...prev, [key]: value }));
  };

  const executeInference = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/v1/predict/mood', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(features),
      });
      
      if (!response.ok) throw new Error('Error en la respuesta de la API');
      
      const data: PredictionResponse = await response.json();
      setPrediction(data.predicted_mood);
    } catch (error) {
      console.error('Fallo en el servicio de predicción:', error);
      setPrediction('Error de conexión');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl border border-neutral-800 p-6 bg-black">
      <div className="space-y-6 mb-8">
        <div>
          <label className="block text-xs uppercase tracking-widest mb-2 font-mono">
            Danceability: {features.danceability.toFixed(2)}
          </label>
          <input
            type="range" min="0" max="1" step="0.01"
            value={features.danceability}
            onChange={(e) => handleSliderChange('danceability', parseFloat(e.target.value))}
            className="w-full accent-current"
          />
        </div>

        <div>
          <label className="block text-xs uppercase tracking-widest mb-2 font-mono">
            Energy: {features.energy.toFixed(2)}
          </label>
          <input
            type="range" min="0" max="1" step="0.01"
            value={features.energy}
            onChange={(e) => handleSliderChange('energy', parseFloat(e.target.value))}
            className="w-full accent-current"
          />
        </div>

        <div>
          <label className="block text-xs uppercase tracking-widest mb-2 font-mono">
            Acousticness: {features.acousticness.toFixed(2)}
          </label>
          <input
            type="range" min="0" max="1" step="0.01"
            value={features.acousticness}
            onChange={(e) => handleSliderChange('acousticness', parseFloat(e.target.value))}
            className="w-full accent-current"
          />
        </div>

        <div>
          <label className="block text-xs uppercase tracking-widest mb-2 font-mono">
            Tempo (BPM): {features.tempo}
          </label>
          <input
            type="range" min="60" max="180" step="1"
            value={features.tempo}
            onChange={(e) => handleSliderChange('tempo', parseInt(e.target.value))}
            className="w-full accent-current"
          />
        </div>
      </div>

      <button
        onClick={executeInference}
        disabled={loading}
        className="w-full border border-neutral-700 bg-neutral-900 hover:bg-neutral-800 text-sm font-mono uppercase tracking-wider py-3 transition-colors disabled:opacity-50"
      >
        {loading ? 'Procesando...' : 'Ejecutar Inferencia'}
      </button>

      {prediction && (
        <div className="mt-8 pt-6 border-t border-neutral-800 text-center">
          <span className="block text-xs uppercase tracking-widest text-neutral-400 font-mono mb-2">
            Resultado del Modelo
          </span>
          <span className="text-4xl font-black uppercase tracking-tight block">
            {prediction}
          </span>
        </div>
      )}
    </div>
  );
}