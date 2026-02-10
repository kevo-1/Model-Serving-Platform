package http

import (
	"encoding/json"
	"net/http"
)

type ModelsResponse struct {
    Models []string `json:"models"`
    Count  int      `json:"count"`
}

func (h *Handler) handleListModels(w http.ResponseWriter, r *http.Request) {

	models := h.modelRegistry.List()

	response := ModelsResponse{
		Models: models,
		Count: len(models),
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}