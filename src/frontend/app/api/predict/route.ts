export async function POST(request) {
  try {
    const body = await request.json();
    const res = await fetch('http://localhost:5000/api/predict', {
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