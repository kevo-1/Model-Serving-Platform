package domain

import (
	"fmt"
)

type ModelNotFoundError struct {
	ModelID string
}

func (e *ModelNotFoundError) Error() string {
	return fmt.Sprintf("model not found: %s", e.ModelID)
}

type InvalidInputError struct {
	Expected int
	Got      int
}

func (e *InvalidInputError) Error() string {
	return fmt.Sprintf("invalid input: expected %d features, got %d", e.Expected, e.Got)
}

type PredictionError struct {
	ModelID string
	Cause   error
}

func (e *PredictionError) Error() string {
	return fmt.Sprintf("prediction failed for model %s: %v", e.ModelID, e.Cause)
}

type ModelAlreadyExistsError struct {
	ModelID string
}

func (e *ModelAlreadyExistsError) Error() string {
	return fmt.Sprintf("model already exists: %s", e.ModelID)
}
