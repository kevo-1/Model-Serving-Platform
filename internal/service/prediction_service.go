package service

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/kevo-1/model-serving-platform/internal/domain"
	"github.com/kevo-1/model-serving-platform/internal/repository"
)

type PredictionService struct {
	registry *repository.ModelRegistry
}

func NewPredictionService(registry *repository.ModelRegistry) *PredictionService {
    return &PredictionService{
        registry: registry,
    }
}

func (s *PredictionService) Predict(ctx context.Context, req domain.PredictionRequest) (domain.PredictionResponse, error) {
    //validate request
	if err := req.Validate();err != nil {
		return domain.PredictionResponse{}, err
	}
	
    //generate RequestID if empty
	if req.RequestID == "" {
		req.RequestID = uuid.New().String()
	}

    //Start timing
	start := time.Now()
    
    //get model from registry
	model, err := s.registry.Get(req.ModelID)
	if err != nil {
		return domain.PredictionResponse{}, err
	}

    //call model.Predict()
	prediction, err := model.Predict(ctx, req.Features)
	if err != nil {
		return domain.PredictionResponse{}, err
	}

    //Build response with timing
	latency := float64(time.Since(start).Microseconds())/1000
	response := domain.PredictionResponse{
		ModelID: req.ModelID,
		RequestID: req.RequestID,
		LatencyMs: latency,
		Prediction: prediction,
		Timestamp: time.Now(),
	}

	return response, nil
}