import { API_BASE_URL } from "@/lib/utils";

export async function GET() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/demo-examples`);
    const data = await res.json();
    return Response.json(data);
  } catch (error) {
    return Response.json({ error: 'Error fetching demo examples from backend', details: String(error) }, { status: 500 });
  }
}
