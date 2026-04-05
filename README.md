# Model Nexus

A machine learning inference API built with Go for dynamically serving ONNX models with Prometheus metrics and structured logging.

## Features

### Core Functionality
- **Dynamic ONNX Model Upload** — `POST /models/upload` accepts arbitrary ONNX files at runtime with automatic metadata extraction
- **Pure-Go Protobuf Parser** — Extracts model metadata (inputs, outputs, dtypes, shapes) without Python dependencies using raw `protowire` decoding
- **ONNX Runtime Inference** — Real-time predictions with full dtype support (float32, float64, int32, int64)
- **Thread-Safe Model Registry** — Concurrent-safe model storage with `sync.RWMutex`

### Observability
- **Prometheus Metrics** — HTTP request counts/latency, prediction counts, error rates, models loaded
- **Structured Logging** — JSON logs with request ID tracing via `slog`
- **Request IDs** — Auto-generated UUID per request, propagated to all logs and response headers

### Production Ready
- **Graceful Shutdown** — Drains in-flight requests with 15s timeout on SIGTERM
- **Docker Support** — Multi-stage builds with CGO for ONNX Runtime
- **Railway Deployment** — Live production deployment [here](https://model-nexus.up.railway.app)
- **Health Checks** — `/health` endpoint for monitoring
- **CORS Enabled** — Ready for web frontends

## Architecture

Built using **Clean Architecture** principles with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Request                          │
├─────────────────────────────────────────────────────────┤
│  Middleware Stack: CORS → RequestID → Metrics            │
├─────────────────────────────────────────────────────────┤
│  Handler Layer     → Request validation & response       │
│  Service Layer     → Business logic (PredictionService)  │
│  Repository Layer  → Model registry (in-memory map)      │
│  ONNX Predictor    → Inference via ONNX Runtime          │
├─────────────────────────────────────────────────────────┤
│                    JSON Response                         │
└─────────────────────────────────────────────────────────┘
```

**Key Patterns:**
- Repository Pattern for model management
- Dependency Injection throughout
- Interface-based design for testability
- Sidecar `.model_info.json` files for metadata persistence

## Tech Stack

- **Language:** Go 1.24
- **ML Runtime:** ONNX Runtime (CGO)
- **Protobuf Parsing:** `google.golang.org/protobuf/encoding/protowire` (pure-Go, no Python)
- **Observability:** Prometheus, structured logging (slog)
- **Deployment:** Docker, Railway

## Getting Started

### Prerequisites

- Go 1.24 or higher
- ONNX Runtime library (see installation below)

### Installing ONNX Runtime

**Linux:**
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.1/onnxruntime-linux-x64-1.24.1.tgz
tar -xzf onnxruntime-linux-x64-1.24.1.tgz
sudo cp onnxruntime-linux-x64-1.24.1/lib/libonnxruntime.so.1.24.1 /usr/lib/libonnxruntime.so
```

**macOS:**
```bash
brew install onnxruntime
```

**Windows:**
1. Download `onnxruntime-win-x64-1.24.1.zip` from [releases](https://github.com/microsoft/onnxruntime/releases)
2. Extract to `C:\onnxruntime\`
3. Set `ONNX_LIBRARY_PATH=C:\onnxruntime\lib\onnxruntime.dll`

### Installation

```bash
git clone https://github.com/kevo-1/model-nexus.git
cd model-nexus
go mod download
```

### Configuration

Create a `.env` file (optional):

```bash
ONNX_LIBRARY_PATH=/usr/lib/libonnxruntime.so
PORT=8080
```

### Running

```bash
# Build
go build -o server ./cmd/server

# Run
./server
```

Server starts on `http://localhost:8080` with dynamic model upload enabled.

## API Endpoints

### Upload Model

**POST** `/models/upload`

Upload an ONNX model file for dynamic serving.

**Request:** `multipart/form-data`
- `file` — `.onnx` model file (required)
- `id` — Unique model identifier (required)
- `name` — Human-readable model name (required)
- `version` — Model version string (required)

**Response (201 Created):**
```json
{
  "model": {
    "id": "my_classifier",
    "name": "My Custom Classifier",
    "version": "v1.0.0",
    "path": "models/my_classifier.onnx"
  },
  "info": {
    "inputs": [
      {
        "name": "input",
        "dtype": 1,
        "shape": [1, 4]
      }
    ],
    "outputs": [
      {
        "name": "output",
        "dtype": 7,
        "shape": [1, 3]
      }
    ]
  }
}
```

**Error Responses:**
- `400 Bad Request` — Missing fields or invalid file
- `409 Conflict` — Model ID already registered
- `500 Internal Server Error` — Failed to parse or load model

**Example:**
```bash
curl -X POST http://localhost:8080/models/upload \
  -F "file=@my_model.onnx" \
  -F "id=my_classifier" \
  -F "name=My Classifier" \
  -F "version=v1.0.0"
```

---

### Prediction

**POST** `/predict`

Make a prediction using a registered model.

**Request:**
```json
{
  "model_id": "my_classifier",
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response (200 OK):**
```json
{
  "model_id": "my_classifier",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "prediction": [0.0],
  "latency_ms": 12.5,
  "timestamp": "2026-04-05T10:30:00Z"
}
```

**Error Responses:**
- `400 Bad Request` — Invalid input (wrong feature count, invalid JSON)
- `404 Not Found` — Model not found
- `500 Internal Server Error` — Prediction failed

**Example:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my_classifier", "features": [5.1, 3.5, 1.4, 0.2]}'
```

---

### Health Check

**GET** `/health`

**Response (200 OK):**
```json
{
  "status": "healthy"
}
```

---

### List Models

**GET** `/models`

List all registered models.

**Response (200 OK):**
```json
{
  "models": [
    {
      "id": "my_classifier",
      "name": "My Classifier",
      "version": "v1.0.0",
      "path": "models/my_classifier.onnx"
    }
  ]
}
```

---

### Model Info

**GET** `/models/info?id=<model_id>`

Get detailed metadata for a specific model.

**Response (200 OK):**
```json
{
  "id": "my_classifier",
  "name": "My Classifier",
  "version": "v1.0.0",
  "inputs": [
    {"name": "input", "dtype": 1, "shape": [1, 4]}
  ],
  "outputs": [
    {"name": "output", "dtype": 7, "shape": [1, 3]}
  ]
}
```

---

### Metrics

**GET** `/metrics`

Prometheus metrics endpoint.

**Key Metrics:**
- `http_requests_total` — Total HTTP requests by endpoint and status
- `http_request_duration_seconds` — Request latency histogram
- `model_predictions_total` — Total predictions by model and status
- `model_inference_duration_seconds` — Model inference latency histogram
- `models_loaded` — Number of models currently loaded

## Project Structure

```
├── cmd/server/main.go          # Application entry point
├── internal/
│   ├── domain/                  # Core types, interfaces, errors
│   ├── handler/http/            # HTTP handlers, middleware, routes
│   ├── logger/                  # Structured logging (slog)
│   ├── metrics/                 # Prometheus metrics
│   ├── repository/              # Model registry (in-memory)
│   └── service/                 # Business logic (prediction, model upload)
├── pkg/onnx/
│   ├── onnx_parser.go           # Pure-Go protobuf metadata extractor
│   ├── onnx_predictor.go        # ONNX Runtime integration
│   ├── predictor.go             # ModelPredictor interface
│   └── model_metadata.go        # ModelInfo, TensorInfo types
├── models/                      # Uploaded .onnx files stored here
├── Dockerfile
├── index.html                   # Web frontend
└── scripts/                     # Model training utilities
```

## Monitoring & Observability

### Structured Logging

All logs are JSON with request tracing:

```json
{
  "time": "2026-04-05T10:30:00Z",
  "level": "INFO",
  "msg": "prediction completed",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_id": "my_classifier",
  "latency_ms": 12.5,
  "status": "success"
}
```

### Request Tracing

Every request gets a unique `request_id` in:
- Response header: `X-Request-ID`
- All log entries
- Enables end-to-end request tracing across the stack

## Design Decisions

### Pure-Go ONNX Metadata Extraction

The `pkg/onnx/onnx_parser.go` file implements a raw protobuf parser using `protowire` to extract model metadata (inputs, outputs, dtypes, shapes) without Python or generated proto structs.

**Why?** Avoiding a Python runtime in the Docker image keeps it minimal and Go-only. The trade-off is parsing fragility — if the ONNX spec changes field numbers, this parser could break silently.

**Limitations:**
- Only extracts metadata needed for inference, not full model validation
- Assumes ONNX v1.0+ spec field numbers from [onnx.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto)
- If parsing fails, use the official ONNX Python package to inspect model metadata

## Future Enhancements

- Model drift detection integration
- Accuracy and confidence measuring
- Model versioning and A/B testing
- Batch prediction support
- gRPC support for high-performance scenarios
- E2E test suite with Playwright

## License

MIT
