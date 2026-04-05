package http

import (
	"encoding/json"
	"net/http"
)

type ModelDetail struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Version string `json:"version"`
	Path    string `json:"path"`
}

type ModelsResponse struct {
	Models []ModelDetail `json:"models"`
	Count  int           `json:"count"`
}

func (h *Handler) handleListModels(w http.ResponseWriter, r *http.Request) {
	modelIDs := h.modelRegistry.List()

	details := make([]ModelDetail, 0, len(modelIDs))
	for _, id := range modelIDs {
		predictor, err := h.modelRegistry.Get(id)
		if err != nil {
			continue
		}
		meta := predictor.Metadata()
		details = append(details, ModelDetail{
			ID:      meta.ID,
			Name:    meta.Name,
			Version: meta.Version,
			Path:    meta.Path,
		})
	}

	response := ModelsResponse{
		Models: details,
		Count:  len(details),
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}
