package http

import (
    "net/http"
    "github.com/kevo-1/model-serving-platform/internal/service"
    "github.com/kevo-1/model-serving-platform/internal/repository"
)

type Handler struct {
	predictionService *service.PredictionService
    modelRegistry     *repository.ModelRegistry
}

func NewHandler(registery *repository.ModelRegistry) *Handler {
	return &Handler{
		predictionService: service.NewPredictionService(registery),
        modelRegistry: registery,
	}
}

func (h *Handler) SetupRoutes() http.Handler {
    mux := http.NewServeMux()
    
    mux.HandleFunc("/predict", h.handlePredict)
    mux.HandleFunc("/health", h.handleHealth)
    mux.HandleFunc("/models", h.handleListModels)
    
    return corsMiddleware(mux)
}

func corsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
        
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        
        next.ServeHTTP(w, r)
    })
}