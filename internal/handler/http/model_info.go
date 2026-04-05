package http

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/kevo-1/model-nexus/internal/logger"
	"github.com/kevo-1/model-nexus/pkg/onnx"
)

type ModelInfoResponse struct {
	ID      string            `json:"id"`
	Name    string            `json:"name"`
	Version string            `json:"version"`
	Inputs  []onnx.TensorInfo `json:"inputs"`
	Outputs []onnx.TensorInfo `json:"outputs"`
}

func (h *Handler) handleModelInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract model ID from path: /models/info?id=xxx
	modelID := r.URL.Query().Get("id")
	if modelID == "" {
		// Fallback: try path-based extraction /models/{id}/info
		path := strings.TrimPrefix(r.URL.Path, "/models/")
		path = strings.TrimSuffix(path, "/info")
		modelID = path
	}

	if modelID == "" {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "model ID is required"})
		return
	}

	predictor, err := h.modelRegistry.Get(modelID)
	if err != nil {
		logger.Warn("model info: model not found", "model_id", modelID, "error", err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	meta := predictor.Metadata()

	resp := ModelInfoResponse{
		ID:      meta.ID,
		Name:    meta.Name,
		Version: meta.Version,
	}

	// Extract model info if available
	type InfoProvider interface {
		ModelInfo() *onnx.ModelInfo
	}

	if ip, ok := predictor.(InfoProvider); ok {
		info := ip.ModelInfo()
		if info != nil {
			resp.Inputs = info.Inputs
			resp.Outputs = info.Outputs
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
