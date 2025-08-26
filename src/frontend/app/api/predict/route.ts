import { API_BASE_URL } from "@/lib/utils";
import type { NextRequest } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    return Response.json(data);
  } catch (error) {
    return Response.json({ error: 'Error fetching prediction from backend', details: String(error) }, { status: 500 });
  }
}