package onnx

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type ONNXDtype int

const (
	DtypeFloat  ONNXDtype = 1
	DtypeUint8  ONNXDtype = 2
	DtypeInt8   ONNXDtype = 3
	DtypeUint16 ONNXDtype = 4
	DtypeInt16  ONNXDtype = 5
	DtypeInt32  ONNXDtype = 6
	DtypeInt64  ONNXDtype = 7
	DtypeString ONNXDtype = 8
	DtypeBool   ONNXDtype = 9
	DtypeDouble ONNXDtype = 11
)

type TensorInfo struct {
	Name  string    `json:"name"`
	Shape []int64   `json:"shape"`
	Dtype ONNXDtype `json:"dtype"`
}

type ModelInfo struct {
	Inputs  []TensorInfo `json:"inputs"`
	Outputs []TensorInfo `json:"outputs"`
}

func LoadModelInfo(modelPath string) (*ModelInfo, error) {
	infoPath := strings.TrimSuffix(modelPath, ".onnx") + ".model_info.json"

	data, err := os.ReadFile(infoPath)
	if err != nil {
		return nil, fmt.Errorf("model info sidecar not found at %s: %w", infoPath, err)
	}

	var info ModelInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return nil, fmt.Errorf("failed to parse model info: %w", err)
	}

	if len(info.Inputs) == 0 {
		return nil, fmt.Errorf("model info has no inputs defined")
	}
	if len(info.Outputs) == 0 {
		return nil, fmt.Errorf("model info has no outputs defined")
	}

	return &info, nil
}

func (m *ModelInfo) InputSize() int {
	shape := m.Inputs[0].Shape
	size := int64(1)
	hasPositiveDim := false
	for _, dim := range shape {
		if dim > 0 {
			size *= dim
			hasPositiveDim = true
		}
	}
	if !hasPositiveDim {
		return -1
	}
	return int(size)
}

func (m *ModelInfo) InputNames() []string {
	names := make([]string, len(m.Inputs))
	for i, input := range m.Inputs {
		names[i] = input.Name
	}
	return names
}

func (m *ModelInfo) OutputNames() []string {
	names := make([]string, len(m.Outputs))
	for i, output := range m.Outputs {
		names[i] = output.Name
	}
	return names
}
