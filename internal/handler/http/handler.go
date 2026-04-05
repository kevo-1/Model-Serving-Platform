package http

import (
	"net/http"

	"github.com/kevo-1/model-nexus/internal/repository"
	"github.com/kevo-1/model-nexus/internal/service"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Handler struct {
	predictionService *service.PredictionService
	modelService      *service.ModelService
	modelRegistry     *repository.ModelRegistry
}

func NewHandler(registry *repository.ModelRegistry, modelsDir string) *Handler {
	return &Handler{
		predictionService: service.NewPredictionService(registry),
		modelService:      service.NewModelService(registry, modelsDir),
		modelRegistry:     registry,
	}
}

func (h *Handler) SetupRoutes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/predict", h.handlePredict)
	mux.HandleFunc("/health", h.handleHealth)
	mux.HandleFunc("/models/upload", h.handleUploadModel)
	mux.HandleFunc("/models/info", h.handleModelInfo)
	mux.HandleFunc("/models", h.handleListModels)
	mux.Handle("/metrics", promhttp.Handler())

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		http.ServeFile(w, r, "index.html")
	})

	handler := corsMiddleware(mux)
	handler = RequestIDMiddleware(handler)
	handler = MetricsMiddleware(handler)

	return handler
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
