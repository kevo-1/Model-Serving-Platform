# Model Nexus

A machine learning inference API built with Go for registering, deploying, and serving ONNX models with solid metrics and logs.

## Features

### Core Functionality
- **ONNX Model Inference** - Real-time predictions with ONNX Runtime
- **Model Registry** - Thread-safe model storage and retrieval
- **Concurrent Request Handling** - Safe for high-load scenarios

### Observability
- **Prometheus Metrics** - HTTP latency, prediction counts, error rates
- **Structured Logging** - JSON logs with request tracing
- **Request IDs** - Trace requests across the entire stack

### Production Ready
- **Docker Support** - Multi-stage builds with CGO
- **Railway Deployment** - Live production deployment [here](https://model-nexus.up.railway.app)
- **Health Checks** - Endpoint for monitoring
- **CORS Enabled** - Ready for web frontends

## Architecture

Built using **Clean Architecture** principles with clear separation of concerns:

- **Domain Layer**: Core business logic, interfaces, and error types
- **Service Layer**: Business logic orchestration (PredictionService)
- **Handler Layer**: HTTP request/response handling
- **Repository Layer**: Model registry and storage
- **Infrastructure**: ONNX runtime integration, metrics, logging

**Request Flow:**
```
HTTP Request → Middleware (CORS, RequestID, Metrics) 
  → Handler (validation, parsing)
  → Service (business logic)
  → Repository (model lookup)
  → ONNX Predictor (inference)
  → Response
```

**Key Patterns:**
- Repository Pattern for model management
- Dependency Injection
- Interface-based design for testability

## Tech Stack

- **Language:** Go 1.24
- **ML Runtime:** ONNX Runtime
- **Observability:** Prometheus, structured logging (slog)
- **Deployment:** Docker, Railway

## Getting Started

### Prerequisites

- Go 1.24 or higher
- ONNX Runtime library

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
3. Add to PATH or set `ONNX_LIBRARY_PATH` environment variable

### Installation

```bash
# Clone the repository
git clone https://github.com/kevo-1/model-nexus.git
cd model-nexus

# Install dependencies
go mod download

# Build
go build ./cmd/server
```

### Configuration

Create a `.env` file (optional):

```bash
# ONNX Runtime library path
ONNX_LIBRARY_PATH=/usr/lib/libonnxruntime.so

# Server port (default: 8080)
PORT=8080
```

### Running

```bash
# With environment variable
export ONNX_LIBRARY_PATH=/usr/lib/libonnxruntime.so
./server

# Or on Windows
set ONNX_LIBRARY_PATH=C:\onnxruntime\lib\onnxruntime.dll
server.exe
```

Server will start on `http://localhost:8080`

## API Endpoints

### Prediction Endpoint

**POST** `/predict`

Make a prediction using a registered model.

**Request:**
```json
{
  "model_id": "iris_v1",
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response (200 OK):**
```json
{
  "model_id": "iris_v1",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "prediction": [0.0],
  "latency_ms": 12.5,
  "timestamp": "2026-02-14T10:30:00Z"
}
```

**Error Responses:**
- `400 Bad Request` - Invalid input (wrong feature count, invalid JSON)
- `404 Not Found` - Model not found
- `500 Internal Server Error` - Prediction failed

**Example:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "iris_v1", "features": [5.1, 3.5, 1.4, 0.2]}'
```

---

### Health Check

**GET** `/health`

Check if the server is running.

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
      "id": "iris_v1",
      "name": "Iris Classifier",
      "version": "v1.0.0",
      "path": "models/iris_classifier_v1.onnx"
    }
  ]
}
```

---

### Metrics

**GET** `/metrics`

Prometheus metrics endpoint.

**Response:** Prometheus text format with metrics like:
- `http_requests_total` - Total HTTP requests by endpoint and status
- `model_predictions_total` - Total predictions by model and status
- `model_inference_duration_seconds` - Model inference latency histogram
- `models_loaded` - Number of models currently loaded

## Project Structure

```
Project root
│   Dockerfile
│   go.mod    
│   go.sum    
│   index.html
│
├───cmd
│   └───server
│           main.go
│
├───internal
│   ├───domain
│   │       error.go
│   │       model.go
│   │       prediction.go
│   │
│   ├───handler
│   │   └───http
│   │           handler.go
│   │           health.go
│   │           middleware.go
│   │           models.go
│   │           predict.go
│   │
│   ├───logger
│   │       logger.go
│   │
│   ├───metrics
│   │       metrics.go
│   │
│   ├───repository
│   │       model_registry.go
│   │
│   └───service
│           prediction_service.go
│
├───models
│       .gitkeep
│       iris_classifier_v1.onnx
│
├───pkg
│   └───onnx
│           onnx_predictor.go
│           predictor.go
│
└───scripts
        requirements.txt
        train_model.py
```

## Monitoring & Observability

### Structured Logging

All logs are in JSON format with request tracing:

```json
{
  "time": "2026-02-14T10:30:00Z",
  "level": "INFO",
  "msg": "prediction completed",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_id": "iris_v1",
  "latency_ms": 12.5,
  "status": "success"
}
```

### Prometheus Metrics

Access metrics at `/metrics` endpoint. Key metrics:

- **HTTP Metrics**: Request count, duration histograms
- **Model Metrics**: Prediction count, inference latency
- **System Metrics**: Loaded models count

### Request Tracing

Every request gets a unique `request_id` in:
- Response header: `X-Request-ID`
- All log entries
- Enables end-to-end request tracing



## Development

### Adding a New Model

1. Train your model and export to ONNX format
2. Place in `models/` directory
3. Register in `main.go`:

```go
model, err := onnx.NewONNXPredictor(
    "model_id",
    "Model Name",
    "v1.0.0",
    "models/your_model.onnx",
)
registry.Register("model_id", model)
```

### Project Principles

- **Clean Architecture** - Clear separation of concerns
- **Interface-Driven** - Testable, mockable dependencies
- **Structured Logging** - Observable, traceable requests
- **Type Safety** - Strong typing, minimal `interface{}`

## Future Enhancements

- Model drift detection integration
- Accuracy and confidence measuring
- Dynamic model loading via API
- Model versioning and A/B testing
- Batch prediction support
- gRPC support for high-performance scenarios

## License

MIT