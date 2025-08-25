import { NextResponse } from 'next/server';

function predictFromText(text) {
  const categories = ['cardiovascular', 'neurologico', 'hepatorenal', 'oncologico'];
  
  const keywords = {
    cardiovascular: ['dolor torácico', 'disnea', 'troponinas', 'ecg', 'cardiaco', 'corazón', 'infarto', 'cardiac', 'chest pain'],
    neurologico: ['cefalea', 'consciencia', 'déficit motor', 'tac cerebral', 'neurológico', 'cerebro', 'stroke', 'brain'],
    hepatorenal: ['ictericia', 'ascitis', 'bilirrubinas', 'transaminasas', 'hepático', 'renal', 'riñón', 'liver', 'kidney'],
    oncologico: ['masa', 'adenopatías', 'biopsia', 'adenocarcinoma', 'tumor', 'cancer', 'oncológico', 'malignant']
  };
  
  const results = categories.map(category => {
    const categoryKeywords = keywords[category];
    const matches = categoryKeywords.filter(keyword => 
      text.toLowerCase().includes(keyword.toLowerCase())
    ).length;
    
    const baseProbability = Math.min(matches * 0.3, 0.9);
    const randomNoise = (Math.random() - 0.5) * 0.1;
    const probability = Math.max(0.05, Math.min(0.95, baseProbability + randomNoise));
    
    return {
      category,
      probability: Math.round(probability * 1000) / 1000,
      predicted: probability > 0.5,
      confidence: probability > 0.8 ? 'Alta' : probability > 0.5 ? 'Media' : 'Baja'
    };
  });
  
  return {
    text,
    predictions: results,
    feature_importance: [
      { feature: 'dolor', value: 0.85, importance: 0.12 },
      { feature: 'paciente', value: 0.72, importance: 0.08 },
      { feature: 'torácico', value: 0.91, importance: 0.15 },
      { feature: 'cerebral', value: 0.68, importance: 0.10 },
      { feature: 'hepático', value: 0.73, importance: 0.09 }
    ],
    prediction_summary: {
      total_categories: results.filter(r => r.predicted).length,
      max_probability: Math.max(...results.map(r => r.probability)),
      primary_category: results.reduce((a, b) => a.probability > b.probability ? a : b).category
    },
    timestamp: new Date().toISOString()
  };
}

export async function POST(request) {
  try {
    const { text } = await request.json();
    
    if (!text) {
      return NextResponse.json(
        { error: 'Se requiere campo "text"' }, 
        { status: 400 }
      );
    }
    
    const prediction = predictFromText(text);
    return NextResponse.json(prediction);
    
  } catch (error) {
    return NextResponse.json(
      { error: `Error en predicción: ${error.message}` },
      { status: 500 }
    );
  }
}
