package http

import (
	"encoding/json"
	"net/http"
    "fmt"

	"github.com/kevo-1/model-serving-platform/internal/domain"
	"github.com/kevo-1/model-serving-platform/internal/logger"
)

func (h *Handler) handlePredict(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    requestID := logger.GetRequestID(r.Context())
    
    var req domain.PredictionRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        logger.Warn("json decode error",
            "request_id", requestID,
            "error", err,
        )
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    res, err := h.predictionService.Predict(r.Context(), req)
    
    if err != nil {
        switch e := err.(type) {
            case *domain.ValidationError:
                http.Error(w, e.Error(), http.StatusBadRequest)
                return
            case *domain.ModelNotFoundError:
                http.Error(w, e.Error(), http.StatusNotFound)
                return
            case *domain.InvalidInputError:
                http.Error(w, e.Error(), http.StatusBadRequest)
                return
            default:
                logger.Error("unhandled error type", 
                    "request_id", requestID,
                    "error_type", fmt.Sprintf("%T", err),
                    "error", err,
                )
                http.Error(w, "Internal server error", http.StatusInternalServerError)
                return
        }
    }
    
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(res)
}