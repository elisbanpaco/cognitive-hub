import MoodPredictor from './_components/MoodPredictor';

export default function MoodPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold uppercase tracking-wider mb-2">
        Análisis Predictivo de Audio
      </h1>
      <p className="text-sm tracking-wide mb-8">
        Clasificación automatizada de estados de ánimo utilizando el modelo de producción RandomForest.
      </p>
      <MoodPredictor />
    </div>
  );
}