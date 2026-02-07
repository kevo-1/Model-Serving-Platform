package domain

import (
	"time"
	"fmt"
)

type PredictionRequest struct {
	ModelID string `json:"model_id"`
	RequestID string `json:"request_id"`
	Features []float64 `json:"features"`
}

type PredictionResponse struct {
	ModelID string `json:"model_id"`
	RequestID string `json:"request_id"`
	LatencyMs float64 `json:"latency_ms"`
	Prediction []float64 `json:"prediction"`
	Timestamp   time.Time `json:"timestamp"`
	Confidence  *float64  `json:"confidence,omitempty"`
}


func (req *PredictionRequest) Validate() error {
	if req.ModelID == "" {
		return &ValidationError{Field: "model_id", Message: "model_id is required"}
	}

	if len(req.Features) == 0 {
		return &ValidationError{Field: "features", Message: "features cannot be empty"}
	}

	return nil
}

type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("Validation error [%s]: %s", e.Field, e.Message)
}