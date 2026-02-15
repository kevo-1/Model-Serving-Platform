package service

import (
	"context"
	"time"

	"github.com/kevo-1/model-serving-platform/internal/domain"
	"github.com/kevo-1/model-serving-platform/internal/logger"
	"github.com/kevo-1/model-serving-platform/internal/metrics"
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
	requestID := logger.GetRequestID(ctx)

	// If request came without ID, use the one from context
	if req.RequestID == "" {
		req.RequestID = requestID
	}
    
    //get model from registry
	model, err := s.registry.Get(req.ModelID)
	if err != nil {
		logger.Error("model not found in registry", 
			"request_id", req.RequestID,
			"model_id", req.ModelID,
			"error", err,
		)
		return domain.PredictionResponse{}, err
	}

    
	logger.Info("prediction started", "request_id",req.RequestID, "model_id", req.ModelID)
    inferenceStart := time.Now()
    prediction, err := model.Predict(ctx, req.Features)
    inferenceDuration := time.Since(inferenceStart).Seconds()
    
    // Record metrics
    success := err == nil
    metrics.RecordPrediction(req.ModelID, success, inferenceDuration)
    
    if err != nil {
		logger.Error("prediction failed", 
			"request_id", req.RequestID,
			"model_id", req.ModelID,
			"error", err,
		)
        return domain.PredictionResponse{}, err
    }

	logger.Info("prediction completed", 
		"request_id", req.RequestID,
		"model_id", req.ModelID,
		"latency_ms", inferenceDuration * 1000,
		"status", "success",
	)

    //Build response with timing
	totalLatency := float64(time.Since(inferenceStart).Microseconds()) / 1000
	
	response := domain.PredictionResponse{
		ModelID: req.ModelID,
		RequestID: req.RequestID,
		LatencyMs: totalLatency,
		Prediction: prediction,
		Timestamp: time.Now(),
	}

	return response, nil
}