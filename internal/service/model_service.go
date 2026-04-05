package service

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/kevo-1/model-nexus/internal/domain"
	"github.com/kevo-1/model-nexus/internal/logger"
	"github.com/kevo-1/model-nexus/internal/repository"
	"github.com/kevo-1/model-nexus/pkg/onnx"
)

type ModelService struct {
	registry  *repository.ModelRegistry
	modelsDir string
}

func NewModelService(registry *repository.ModelRegistry, modelsDir string) *ModelService {
	return &ModelService{
		registry:  registry,
		modelsDir: modelsDir,
	}
}

type RegisterModelRequest struct {
	ID      string
	Name    string
	Version string
	File    io.Reader
}

type RegisterModelResponse struct {
	Model domain.ModelMetadata `json:"model"`
	Info  *onnx.ModelInfo      `json:"info"`
}

func (s *ModelService) RegisterModel(req RegisterModelRequest) (*RegisterModelResponse, error) {
	// 1. Validate request fields
	if req.ID == "" || req.Name == "" || req.Version == "" {
		return nil, &domain.ValidationError{
			Field:   "id/name/version",
			Message: "id, name, and version are required",
		}
	}

	// 2. Build file paths
	onnxPath := filepath.Join(s.modelsDir, req.ID+".onnx")
	infoPath := filepath.Join(s.modelsDir, req.ID+".model_info.json")

	// 3. Save the .onnx file to disk
	if err := saveFile(req.File, onnxPath); err != nil {
		return nil, fmt.Errorf("failed to save model file: %w", err)
	}
	logger.Info("model file saved", "path", onnxPath)

	// 4. Extract model info using pure Go protobuf parser — no Python needed
	info, err := onnx.ExtractModelInfo(onnxPath)
	if err != nil {
		// Clean up the saved file if parsing fails
		os.Remove(onnxPath)
		return nil, fmt.Errorf("failed to extract model info: %w", err)
	}

	// 5. Persist the sidecar JSON so LoadModelInfo can read it on restart
	if err := saveModelInfoJSON(info, infoPath); err != nil {
		os.Remove(onnxPath)
		return nil, fmt.Errorf("failed to save model info sidecar: %w", err)
	}
	logger.Info("model info sidecar saved", "path", infoPath)

	// 6. Create the predictor (reads sidecar internally via LoadModelInfo)
	predictor, err := onnx.NewONNXPredictor(req.ID, req.Name, req.Version, onnxPath)
	if err != nil {
		os.Remove(onnxPath)
		os.Remove(infoPath)
		return nil, fmt.Errorf("failed to initialize model predictor: %w", err)
	}

	// 7. Register in the registry
	if err := s.registry.Register(req.ID, predictor); err != nil {
		predictor.Close()
		os.Remove(onnxPath)
		os.Remove(infoPath)
		return nil, err // already typed (ModelAlreadyExistsError)
	}

	logger.Info("model registered successfully",
		"model_id", req.ID,
		"name", req.Name,
		"version", req.Version,
		"inputs", len(info.Inputs),
		"outputs", len(info.Outputs),
	)

	return &RegisterModelResponse{
		Model: predictor.Metadata(),
		Info:  info,
	}, nil
}

func saveFile(src io.Reader, dst string) error {
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := io.Copy(f, src); err != nil {
		return err
	}
	return nil
}

func saveModelInfoJSON(info *onnx.ModelInfo, path string) error {
	data, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
