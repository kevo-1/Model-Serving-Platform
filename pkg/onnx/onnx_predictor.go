package onnx

import (
	"context"
	"fmt"
	"sync"

	"github.com/kevo-1/model-nexus/internal/domain"
	ort "github.com/yalue/onnxruntime_go"
)

type ONNXPredictor struct {
    ID      string
    Name    string
    Path    string
    Version string
    Info    *ModelInfo

    session     *ort.DynamicSession[float64, int64]
    inputNames  []string
    outputNames []string
    inputShape  []int64
    outputShape []int64

    mu sync.Mutex
}

func NewONNXPredictor(id, name, version, path string) (*ONNXPredictor, error) {
    if id == "" || name == "" || path == "" {
        return nil, fmt.Errorf("id, name, and path cannot be empty")
    }

    info, err := LoadModelInfo(path)
    if err != nil {
        return nil, fmt.Errorf("failed to load model info for %s: %w", path, err)
    }

    inputShape := ort.NewShape(info.Inputs[0].Shape...)
    outputShape := ort.NewShape(info.Outputs[0].Shape...)

    for i, d := range inputShape {
        if d == 0 {
            inputShape[i] = 1
        }
    }
    for i, d := range outputShape {
        if d == 0 {
            outputShape[i] = 1
        }
    }

    session, err := ort.NewDynamicSession[float64, int64](
        path,
        info.InputNames(),
        info.OutputNames(),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create ONNX session: %w", err)
    }

    return &ONNXPredictor{
        ID:          id,
        Name:        name,
        Version:     version,
        Path:        path,
        Info:        info,
        session:     session,
        inputNames:  info.InputNames(),
        outputNames: info.OutputNames(),
        inputShape:  inputShape,
        outputShape: outputShape,
    }, nil
}

func (p *ONNXPredictor) Predict(ctx context.Context, features []float64) ([]float64, error) {
    expectedSize := p.Info.InputSize()
    if expectedSize > 0 && len(features) != expectedSize {
        return nil, &domain.InvalidInputError{Expected: expectedSize, Got: len(features)}
    }

    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    default:
    }

    p.mu.Lock()
    defer p.mu.Unlock()

    inputTensor, err := ort.NewTensor(p.inputShape, features)
    if err != nil {
        return nil, &domain.PredictionError{ModelID: p.ID, Cause: err}
    }
    defer inputTensor.Destroy()

    outputTensor, err := ort.NewEmptyTensor[int64](p.outputShape)
    if err != nil {
        return nil, &domain.PredictionError{ModelID: p.ID, Cause: err}
    }
    defer outputTensor.Destroy()

    err = p.session.Run(
        []*ort.Tensor[float64]{inputTensor},
        []*ort.Tensor[int64]{outputTensor},
    )
    if err != nil {
        return nil, &domain.PredictionError{ModelID: p.ID, Cause: err}
    }

    raw := outputTensor.GetData()
    output := make([]float64, len(raw))
    for i, v := range raw {
        output[i] = float64(v)
    }
    return output, nil
}

func (p *ONNXPredictor) Metadata() domain.ModelMetadata {
    return domain.ModelMetadata{
        ID:      p.ID,
        Name:    p.Name,
        Path:    p.Path,
        Version: p.Version,
    }
}

func (p *ONNXPredictor) Close() error {
    if p.session != nil {
        return p.session.Destroy()
    }
    return nil
}