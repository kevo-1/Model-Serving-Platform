package onnx

import (
    "context"
    "math/rand"
    "time"

    "github.com/kevo-1/model-serving-platform/internal/domain"
)

type DummyPredictor struct {
    ID      string
    Name    string
    Path    string
    Version string
}

func NewDummyPredictor(id, name, version, path string) *DummyPredictor {
    return &DummyPredictor{
        ID:      id,
        Name:    name,
        Path:    path,
        Version: version,
    }
}

func (p *DummyPredictor) Predict(ctx context.Context, features []float64) ([]float64, error) {
    if len(features) != 4 {
        return nil, &domain.InvalidInputError{
            Expected: 4,
            Got:      len(features),
        }
    }

    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    default:
    }

    delay := time.Duration(rand.Intn(10)+1) * time.Millisecond
    timer := time.NewTimer(delay)
    
    select {
    case <-timer.C:
    case <-ctx.Done():
        timer.Stop()
        return nil, ctx.Err()
    }

    var class float64
    if features[0] < 5.5 {
        class = 0.0
    } else if features[0] < 6.5 {
        class = 1.0
    } else {
        class = 2.0
    }

    return []float64{class}, nil
}

func (p *DummyPredictor) Metadata() domain.ModelMetadata {
    return domain.ModelMetadata{
        ID:      p.ID,
        Name:    p.Name,
        Version: p.Version,
        Path:    p.Path,
    }
}

func (p *DummyPredictor) Close() error {
    return nil
}