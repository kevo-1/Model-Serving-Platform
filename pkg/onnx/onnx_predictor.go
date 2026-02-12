package onnx

import (
	"fmt"
	"sync"
	"context"

	"github.com/kevo-1/model-serving-platform/internal/domain"
	ort "github.com/yalue/onnxruntime_go"
)

type ONNXPredictor struct {
	ID string
	Name string
	Path string
	Version string

	session *ort.DynamicSession[float32, float32]

	inputName string
	outputName string

	inputShape []int64
	outputShape []int64

	mu sync.Mutex
}


//FIXME: Change the static/hard coded behaviour of extracting model info
func NewONNXPredictor(id, name, version, path string) (*ONNXPredictor, error) {
    if id == "" || name == "" || path == "" {
        return nil, fmt.Errorf("id, name, and path cannot be empty")
    }

    inputShape := ort.NewShape(1, 4)
    outputShape := ort.NewShape(1)

    inputNames := []string{"X"}
    outputNames := []string{"output_label"}

    session, err := ort.NewDynamicSession[float32, float32](
        path,
        inputNames,
        outputNames,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create ONNX session: %w", err)
    }

    return &ONNXPredictor{
        ID:          id,
        Name:        name,
        Version:     version,
        Path:        path,
        session:     session,
        inputName:   inputNames[0],
        outputName:  outputNames[0],
        inputShape:  inputShape,
        outputShape: outputShape,
    }, nil
}


func (p *ONNXPredictor) Predict(ctx context.Context, features []float64) ([]float64, error) {
    if len(features) != 4 {
        return nil, &domain.InvalidInputError{Expected: 4, Got: len(features)}
    }

    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    default:
    }

	p.mu.Lock()
	defer p.mu.Unlock()

    features32 := make([]float32, len(features))
    for i, value := range features {
        features32[i] = float32(value)
    }

    inputTensor, err := ort.NewTensor(p.inputShape, features32)
    if err != nil {
        return nil, &domain.PredictionError{ModelID: p.ID, Cause: err}
    }
    defer inputTensor.Destroy()

    outputTensor, err := ort.NewEmptyTensor[float32](p.outputShape)
    if err != nil {
        return nil, &domain.PredictionError{ModelID: p.ID, Cause: err}
    }
    defer outputTensor.Destroy()

    err = p.session.Run([]*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
    if err != nil {
        return nil, &domain.PredictionError{ModelID: p.ID, Cause: err}
    }

    res := outputTensor.GetData()

    output := make([]float64, len(res))
    for i, value := range res {
        output[i] = float64(value)
    }

    return output, nil
}