# Optional Remote Inference API

The web app can delegate scoring to a lightweight backend when the local device cannot execute the model efficiently. Set `backendEndpoint` when instantiating `AIInferenceController` to enable the remote path.

## Endpoint Summary
- **Method**: `POST`
- **URL**: `/api/emo-aen/infer` (configurable)
- **Content-Type**: `application/json`
- **Authentication**: Recommend short-lived bearer tokens or signed requests; no authentication is enforced by default.

### Request Body
```json
{
  "version": "1.0.0",
  "features": [
    {
      "ruleOfThirdsScore": 0.78,
      "saliencyConfidence": 0.65,
      "horizonAngle": -1.8,
      "horizonConfidence": 0.72,
      "textureStrength": 0.21,
      "balanceRatio": 1.04,
      "cropArea": 0.68,
      "colorHarmony": 0.86,
      "subjectSize": 0.23,
      "leadingLineStrength": 0.34
    }
  ],
  "context": {
    "imageSize": { "width": 2048, "height": 1365 },
    "device": "web",
    "language": "en-US"
  }
}
```

- `version` – semantic version of the expected model. Reject requests for unsupported versions (respond with 409/422).
- `features` – array of candidate feature vectors matching the order produced by `composition_engine.js`.
- `context` – optional metadata for logging/debugging.

### Response Body
```json
{
  "results": [
    {
      "composition": 0.82,
      "aesthetic": 0.79,
      "explanations": {
        "mode": "emo-aen-v1.0.0"
      }
    }
  ]
}
```
- `composition` – normalized composition score (0–1).
- `aesthetic` – auxiliary aesthetic score (0–1).
- `explanations` – optional diagnostic payload (string map).

Return HTTP `503` when the backend is overloaded; the front-end automatically falls back to rule-based scoring.

## Error Codes
- `400 Bad Request` – malformed payload or missing fields.
- `401/403` – authentication failure.
- `409 Conflict` – unsupported model version.
- `500` – unexpected server error.

## Reference Implementation Sketch
```python
@app.post("/api/emo-aen/infer")
def infer(request: InferenceRequest):
    features = np.array([f.to_vector() for f in request.features], dtype=np.float32)
    outputs = ort_session.run(None, {"input": features})
    return {
        "results": [
            {
                "composition": float(outputs[0][i]),
                "aesthetic": float(outputs[1][i])
            }
            for i in range(len(features))
        ]
    }
```

## Deployment Considerations
- Keep latency <300 ms per request by batching on the server.
- Validate inputs to avoid unbounded vectors (impose numeric ranges).
- Log request ID, mode, and latency for observability.
- Rate limit anonymous clients to prevent abuse.
