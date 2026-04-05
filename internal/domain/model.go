package domain

import (
	"context"
)

type ModelPredictor interface {
	Predict(ctx context.Context, features []float64) ([]float64, error)
	Metadata() ModelMetadata
	Close() error
}

type ModelMetadata struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Path    string `json:"path"`
	Version string `json:"version"`
}
