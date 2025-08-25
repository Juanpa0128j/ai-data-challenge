import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    model_loaded: true,
    vectorizer_loaded: true,
    timestamp: new Date().toISOString(),
    environment: 'vercel'
  });
}
