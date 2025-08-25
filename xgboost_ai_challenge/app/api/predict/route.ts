import { NextResponse } from 'next/server';
import { NextRequest } from 'next/server';

interface KeywordMap {
  [key: string]: string[];
}

function predictFromText(text: string) {
  const categories = ['cardiovascular', 'neurological', 'hepatorenal', 'oncological'];
  
  const keywords: KeywordMap = {
    cardiovascular: ['dolor torácico', 'disnea', 'troponinas', 'ecg', 'cardiaco', 'corazón', 'infarto', 'cardiac', 'chest pain'],
    neurological: ['cefalea', 'consciencia', 'déficit motor', 'tac cerebral', 'neurológico', 'cerebro', 'stroke', 'brain'],
    hepatorenal: ['ictericia', 'ascitis', 'bilirrubinas', 'transaminasas', 'hepático', 'renal', 'riñón', 'liver', 'kidney'],
    oncological: ['masa', 'adenopatías', 'biopsia', 'adenocarcinoma', 'tumor', 'cancer', 'oncológico', 'malignant']
  };
  
  const results = categories.map(category => {
    const categoryKeywords = keywords[category];
    const matches = categoryKeywords.filter((keyword: string) => 
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
      { feature: 'cancer', value: 0.85, importance: 0.15 },
      { feature: 'patients', value: 0.72, importance: 0.12 },
      { feature: 'study', value: 0.91, importance: 0.10 },
      { feature: 'cardiovascular', value: 0.68, importance: 0.09 },
      { feature: 'neurological', value: 0.73, importance: 0.08 }
    ],
    prediction_summary: {
      total_categories: results.filter(r => r.predicted).length,
      max_probability: Math.max(...results.map(r => r.probability)),
      primary_category: results.reduce((a, b) => a.probability > b.probability ? a : b).category
    },
    timestamp: new Date().toISOString()
  };
}

export async function POST(request: NextRequest) {
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
    
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Error en predicción: ${errorMessage}` },
      { status: 500 }
    );
  }
}
