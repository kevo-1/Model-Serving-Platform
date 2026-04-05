package onnx

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sync"

	"github.com/kevo-1/model-nexus/internal/domain"
	ort "github.com/yalue/onnxruntime_go"
)

// onnxTypeToORT maps our ONNXDtype to ort.TensorElementDataType and element size.
func onnxTypeToORT(d ONNXDtype) (ort.TensorElementDataType, int) {
	switch d {
	case DtypeFloat:
		return ort.TensorElementDataTypeFloat, 4
	case DtypeDouble:
		return ort.TensorElementDataTypeDouble, 8
	case DtypeInt64:
		return ort.TensorElementDataTypeInt64, 8
	case DtypeInt32:
		return ort.TensorElementDataTypeInt32, 4
	default:
		return ort.TensorElementDataTypeFloat, 4
	}
}

// ── ONNXPredictor ─────────────────────────────────────────────────
type ONNXPredictor struct {
	ID      string
	Name    string
	Path    string
	Version string
	Info    *ModelInfo

	session       *ort.AdvancedSession
	inputTensor   *ort.CustomDataTensor
	inputDtype    ONNXDtype
	inputElemSize int
	outputTensors []*ort.CustomDataTensor
	outputDtypes  []ONNXDtype
	mu            sync.Mutex
}

func NewONNXPredictor(id, name, version, path string) (*ONNXPredictor, error) {
	if id == "" || name == "" || path == "" {
		return nil, fmt.Errorf("id, name, and path cannot be empty")
	}

	info, err := LoadModelInfo(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load model info for %s: %w", path, err)
	}

	if len(info.Inputs) == 0 || len(info.Outputs) == 0 {
		return nil, fmt.Errorf("model must have at least one input and one output")
	}

	// Build input shape (replace 0 dims with 1)
	inputShape := ort.NewShape(info.Inputs[0].Shape...)
	for i, d := range inputShape {
		if d == 0 {
			inputShape[i] = 1
		}
	}

	// Filter outputs to only tensor types (dtype > 0)
	// Skip Map/Sequence/SequenceMap types which have dtype == 0
	type outputMeta struct {
		name   string
		dtype  ONNXDtype
		shape  []int64
		ortIdx int // index into the output tensors list
	}
	var validOutputs []outputMeta
	for _, out := range info.Outputs {
		if out.Dtype == 0 {
			continue // skip non-tensor outputs
		}
		validOutputs = append(validOutputs, outputMeta{
			name:  out.Name,
			dtype: out.Dtype,
			shape: out.Shape,
		})
	}

	if len(validOutputs) == 0 {
		return nil, fmt.Errorf("model has no tensor outputs")
	}

	// Build output shapes
	outputShapes := make([]ort.Shape, len(validOutputs))
	for i, out := range validOutputs {
		s := ort.NewShape(out.shape...)
		for j, d := range s {
			if d == 0 {
				s[j] = 1
			}
		}
		if len(s) == 0 || (len(s) == 1 && s[0] == 0) {
			s = []int64{1}
		}
		outputShapes[i] = s
	}

	// Create output tensors with correct per-output dtype
	inputDtype := info.Inputs[0].Dtype
	inputORTType, inputElemSize := onnxTypeToORT(inputDtype)

	// Count total input elements
	inputTotalElems := int64(1)
	for _, dim := range inputShape {
		inputTotalElems *= dim
	}

	// Create input CustomDataTensor with correct dtype
	inputBytes := make([]byte, int(inputTotalElems)*inputElemSize)
	inputTensor, err := ort.NewCustomDataTensor(inputShape, inputBytes, inputORTType)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// Create output tensors with correct per-output dtype
	outputTensors := make([]*ort.CustomDataTensor, len(validOutputs))
	outputDtypes := make([]ONNXDtype, len(validOutputs))
	outputNames := make([]string, len(validOutputs))
	for i, out := range validOutputs {
		shape := outputShapes[i]
		dtype := out.dtype
		ortType, elemSize := onnxTypeToORT(dtype)
		totalElems := int64(1)
		for _, dim := range shape {
			totalElems *= dim
		}
		buf := make([]byte, int(totalElems)*elemSize)
		t, err := ort.NewCustomDataTensor(shape, buf, ortType)
		if err != nil {
			inputTensor.Destroy()
			cleanupTensors(outputTensors[:i])
			return nil, fmt.Errorf("failed to create output tensor %d: %w", i, err)
		}
		outputTensors[i] = t
		outputDtypes[i] = dtype
		outputNames[i] = out.name
	}

	inputValues := make([]ort.Value, len(info.Inputs))
	for i := range inputValues {
		inputValues[i] = inputTensor
	}

	outputValues := make([]ort.Value, len(outputTensors))
	for i, t := range outputTensors {
		outputValues[i] = t
	}

	session, err := ort.NewAdvancedSession(
		path,
		info.InputNames(),
		outputNames,
		inputValues,
		outputValues,
		nil,
	)
	if err != nil {
		inputTensor.Destroy()
		cleanupTensors(outputTensors)
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &ONNXPredictor{
		ID:            id,
		Name:          name,
		Version:       version,
		Path:          path,
		Info:          info,
		session:       session,
		inputTensor:   inputTensor,
		inputDtype:    inputDtype,
		inputElemSize: inputElemSize,
		outputTensors: outputTensors,
		outputDtypes:  outputDtypes,
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

	// Write input features to the CustomDataTensor byte buffer
	if err := writeFeatures(p.inputTensor.GetData(), features, p.inputDtype); err != nil {
		return nil, &domain.PredictionError{ModelID: p.ID, Cause: err}
	}

	if err := p.session.Run(); err != nil {
		return nil, &domain.PredictionError{ModelID: p.ID, Cause: err}
	}

	// Collect all output values into a flat float64 slice
	var result []float64
	for i, t := range p.outputTensors {
		readFloat64s(t.GetData(), p.outputDtypes[i], &result)
	}

	return result, nil
}

func (p *ONNXPredictor) Metadata() domain.ModelMetadata {
	return domain.ModelMetadata{
		ID:      p.ID,
		Name:    p.Name,
		Path:    p.Path,
		Version: p.Version,
	}
}

func (p *ONNXPredictor) ModelInfo() *ModelInfo {
	return p.Info
}

func (p *ONNXPredictor) Close() error {
	if p.session != nil {
		p.session.Destroy()
	}
	p.inputTensor.Destroy()
	cleanupTensors(p.outputTensors)
	return nil
}

// ── helpers ───────────────────────────────────────────────────────

func writeFeatures(buf []byte, features []float64, dtype ONNXDtype) error {
	switch dtype {
	case DtypeFloat:
		if len(features)*4 != len(buf) {
			return fmt.Errorf("input size mismatch: expected %d values for %d bytes (float32), got %d", len(buf)/4, len(buf), len(features))
		}
		for i, v := range features {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(v)))
		}

	case DtypeDouble:
		if len(features)*8 != len(buf) {
			return fmt.Errorf("input size mismatch: expected %d values for %d bytes (float64), got %d", len(buf)/8, len(buf), len(features))
		}
		for i, v := range features {
			binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
		}

	case DtypeInt64:
		if len(features)*8 != len(buf) {
			return fmt.Errorf("input size mismatch")
		}
		for i, v := range features {
			binary.LittleEndian.PutUint64(buf[i*8:], uint64(int64(v)))
		}

	default:
		return fmt.Errorf("unsupported input dtype: %d", dtype)
	}

	return nil
}

func readFloat64s(data []byte, dtype ONNXDtype, out *[]float64) {
	switch dtype {
	case DtypeFloat:
		for i := 0; i+3 < len(data); i += 4 {
			bits := binary.LittleEndian.Uint32(data[i:])
			f := math.Float32frombits(bits)
			*out = append(*out, float64(f))
		}
	case DtypeDouble:
		for i := 0; i+7 < len(data); i += 8 {
			bits := binary.LittleEndian.Uint64(data[i:])
			f := math.Float64frombits(bits)
			*out = append(*out, f)
		}
	case DtypeInt64:
		for i := 0; i+7 < len(data); i += 8 {
			bits := binary.LittleEndian.Uint64(data[i:])
			*out = append(*out, float64(int64(bits)))
		}
	case DtypeInt32:
		for i := 0; i+3 < len(data); i += 4 {
			bits := binary.LittleEndian.Uint32(data[i:])
			*out = append(*out, float64(int32(bits)))
		}
	default:
		// Fallback: treat as float32
		for i := 0; i+3 < len(data); i += 4 {
			bits := binary.LittleEndian.Uint32(data[i:])
			f := math.Float32frombits(bits)
			*out = append(*out, float64(f))
		}
	}
}

func cleanupTensors(tensors []*ort.CustomDataTensor) {
	for _, t := range tensors {
		if t != nil {
			t.Destroy()
		}
	}
}
