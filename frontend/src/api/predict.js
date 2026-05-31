const API_URL = 'http://localhost:8000/api/predict'

export async function predictLetter(landmarks) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      landmarks: landmarks.map(p => [p.x, p.y, p.z]),
    }),
    signal: AbortSignal.timeout(500),
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json() // { letter: string | null, confidence: number }
}
