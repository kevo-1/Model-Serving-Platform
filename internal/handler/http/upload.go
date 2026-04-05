package http

import (
	"encoding/json"
	"net/http"

	"github.com/kevo-1/model-nexus/internal/domain"
	"github.com/kevo-1/model-nexus/internal/logger"
	"github.com/kevo-1/model-nexus/internal/service"
)

const maxUploadSize = 500 << 20 // 500 MB

func (h *Handler) handleUploadModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	requestID := logger.GetRequestID(r.Context())

	// Limit request body size before parsing
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)

	if err := r.ParseMultipartForm(32 << 20); err != nil { // 32MB in memory, rest on disk
		logger.Warn("failed to parse multipart form", "request_id", requestID, "error", err)
		http.Error(w, "Failed to parse form: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Extract metadata fields from form
	modelID := r.FormValue("id")
	modelName := r.FormValue("name")
	modelVersion := r.FormValue("version")

	if modelID == "" || modelName == "" || modelVersion == "" {
		http.Error(w, "id, name, and version form fields are required", http.StatusBadRequest)
		return
	}

	// Extract the file
	file, header, err := r.FormFile("model")
	if err != nil {
		logger.Warn("failed to get model file from form", "request_id", requestID, "error", err)
		http.Error(w, "model file is required (form field: 'model')", http.StatusBadRequest)
		return
	}
	defer file.Close()

	logger.Info("model upload received",
		"request_id", requestID,
		"model_id", modelID,
		"filename", header.Filename,
		"size_bytes", header.Size,
	)

	res, err := h.modelService.RegisterModel(service.RegisterModelRequest{
		ID:      modelID,
		Name:    modelName,
		Version: modelVersion,
		File:    file,
	})

	if err != nil {
		switch err.(type) {
		case *domain.ValidationError:
			http.Error(w, err.Error(), http.StatusBadRequest)
		case *domain.ModelAlreadyExistsError:
			http.Error(w, err.Error(), http.StatusConflict)
		default:
			logger.Error("model registration failed",
				"request_id", requestID,
				"model_id", modelID,
				"error", err,
			)
			http.Error(w, "Failed to register model: "+err.Error(), http.StatusInternalServerError)
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(res)
}
