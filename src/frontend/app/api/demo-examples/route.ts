export async function GET() {
  try {
    const res = await fetch('http://localhost:5000/api/demo-examples');
    const data = await res.json();
    return Response.json(data);
  } catch (error) {
    return Response.json({ error: 'Error fetching demo examples from backend', details: String(error) }, { status: 500 });
  }
}
