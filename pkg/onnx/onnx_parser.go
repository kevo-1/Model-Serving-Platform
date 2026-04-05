package onnx

import (
	"fmt"
	"os"

	"google.golang.org/protobuf/encoding/protowire"
)

// Field numbers from the ONNX protobuf spec:
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
//
// ModelProto:
//   field 7 = graph (GraphProto)
//
// GraphProto:
//   field 11 = input  (repeated ValueInfoProto)
//   field 12 = output (repeated ValueInfoProto)
//   field 14 = initializer (repeated TensorProto) — used to filter inputs
//
// ValueInfoProto:
//   field 1 = name   (string)
//   field 2 = type   (TypeProto)
//
// TypeProto:
//   field 1 = tensor_type (TypeProto_Tensor)
//
// TypeProto_Tensor:
//   field 1 = elem_type (int32)  — the dtype
//   field 2 = shape     (TensorShapeProto)
//
// TensorShapeProto:
//   field 1 = dim (repeated TensorShapeProto_Dimension)
//
// TensorShapeProto_Dimension:
//   field 1 = dim_value (int64)
//
// TensorProto (initializer):
//   field 8 = name (string)

const (
	modelFieldGraph       = 7
	graphFieldInput       = 11
	graphFieldOutput      = 12
	graphFieldInitializer = 14

	valueInfoFieldName = 1
	valueInfoFieldType = 2

	typeProtoFieldTensor = 1

	tensorTypeFieldElemType = 1
	tensorTypeFieldShape    = 2

	shapeFieldDim    = 1
	dimFieldDimValue = 1

	initializerFieldName = 8
)

// ExtractModelInfo parses an ONNX file and returns ModelInfo without
// any Python or external dependency. It uses raw protowire decoding.
func ExtractModelInfo(modelPath string) (*ModelInfo, error) {
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}

	graphBytes, err := extractField(data, modelFieldGraph)
	if err != nil {
		return nil, fmt.Errorf("failed to extract graph from model: %w", err)
	}

	// Collect initializer names — these are weight tensors, not real inputs
	initNames, err := extractInitializerNames(graphBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to extract initializer names: %w", err)
	}

	inputs, err := extractValueInfos(graphBytes, graphFieldInput, initNames)
	if err != nil {
		return nil, fmt.Errorf("failed to extract inputs: %w", err)
	}

	outputs, err := extractValueInfos(graphBytes, graphFieldOutput, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to extract outputs: %w", err)
	}

	if len(inputs) == 0 {
		return nil, fmt.Errorf("model has no inputs")
	}
	if len(outputs) == 0 {
		return nil, fmt.Errorf("model has no outputs")
	}

	return &ModelInfo{
		Inputs:  inputs,
		Outputs: outputs,
	}, nil
}

// extractField finds the first occurrence of a length-delimited field
// with the given field number and returns its bytes.
func extractField(data []byte, targetField protowire.Number) ([]byte, error) {
	for len(data) > 0 {
		num, typ, n := protowire.ConsumeTag(data)
		if n < 0 {
			return nil, fmt.Errorf("invalid protobuf tag")
		}
		data = data[n:]

		switch typ {
		case protowire.BytesType:
			val, n := protowire.ConsumeBytes(data)
			if n < 0 {
				return nil, fmt.Errorf("invalid bytes field")
			}
			if num == targetField {
				return val, nil
			}
			data = data[n:]

		case protowire.VarintType:
			_, n := protowire.ConsumeVarint(data)
			if n < 0 {
				return nil, fmt.Errorf("invalid varint")
			}
			data = data[n:]

		case protowire.Fixed32Type:
			_, n := protowire.ConsumeFixed32(data)
			if n < 0 {
				return nil, fmt.Errorf("invalid fixed32")
			}
			data = data[n:]

		case protowire.Fixed64Type:
			_, n := protowire.ConsumeFixed64(data)
			if n < 0 {
				return nil, fmt.Errorf("invalid fixed64")
			}
			data = data[n:]

		default:
			return nil, fmt.Errorf("unknown wire type %d for field %d", typ, num)
		}
	}
	return nil, fmt.Errorf("field %d not found", targetField)
}

// extractInitializerNames collects all initializer (weight tensor) names
// from the graph so we can exclude them from the inputs list.
func extractInitializerNames(graphData []byte) (map[string]bool, error) {
	names := make(map[string]bool)
	data := graphData

	for len(data) > 0 {
		num, typ, n := protowire.ConsumeTag(data)
		if n < 0 {
			return nil, fmt.Errorf("invalid tag in graph")
		}
		data = data[n:]

		if typ == protowire.BytesType {
			val, n := protowire.ConsumeBytes(data)
			if n < 0 {
				return nil, fmt.Errorf("invalid bytes in graph")
			}
			if num == graphFieldInitializer {
				name, err := extractStringField(val, initializerFieldName)
				if err == nil && name != "" {
					names[name] = true
				}
			}
			data = data[n:]
		} else {
			n := protowire.ConsumeFieldValue(num, typ, data)
			if n < 0 {
				return nil, fmt.Errorf("invalid field value")
			}
			data = data[n:]
		}
	}

	return names, nil
}

// extractValueInfos pulls all ValueInfoProto messages for a given field
// (input=11 or output=12), skipping any names found in excludeNames.
func extractValueInfos(graphData []byte, fieldNum protowire.Number, excludeNames map[string]bool) ([]TensorInfo, error) {
	var results []TensorInfo
	data := graphData

	for len(data) > 0 {
		num, typ, n := protowire.ConsumeTag(data)
		if n < 0 {
			return nil, fmt.Errorf("invalid tag")
		}
		data = data[n:]

		if typ == protowire.BytesType {
			val, n := protowire.ConsumeBytes(data)
			if n < 0 {
				return nil, fmt.Errorf("invalid bytes")
			}
			if num == fieldNum {
				info, err := parseValueInfo(val)
				if err != nil {
					return nil, err
				}
				if excludeNames == nil || !excludeNames[info.Name] {
					results = append(results, info)
				}
			}
			data = data[n:]
		} else {
			n := protowire.ConsumeFieldValue(num, typ, data)
			if n < 0 {
				return nil, fmt.Errorf("invalid field value")
			}
			data = data[n:]
		}
	}

	return results, nil
}

// parseValueInfo decodes a ValueInfoProto into a TensorInfo.
func parseValueInfo(data []byte) (TensorInfo, error) {
	var info TensorInfo

	name, err := extractStringField(data, valueInfoFieldName)
	if err != nil {
		return info, fmt.Errorf("failed to read value info name: %w", err)
	}
	info.Name = name

	typeBytes, err := extractField(data, valueInfoFieldType)
	if err != nil {
		// Output tensors sometimes omit type info — treat as unknown
		return info, nil
	}

	tensorTypeBytes, err := extractField(typeBytes, typeProtoFieldTensor)
	if err != nil {
		return info, nil
	}

	dtype, err := extractVarintField(tensorTypeBytes, tensorTypeFieldElemType)
	if err == nil {
		info.Dtype = ONNXDtype(dtype)
	}

	shapeBytes, err := extractField(tensorTypeBytes, tensorTypeFieldShape)
	if err == nil {
		dims, err := extractDims(shapeBytes)
		if err == nil {
			info.Shape = dims
		}
	}

	return info, nil
}

// extractDims reads all dimension values from a TensorShapeProto.
func extractDims(shapeData []byte) ([]int64, error) {
	var dims []int64

	for len(shapeData) > 0 {
		num, typ, n := protowire.ConsumeTag(shapeData)
		if n < 0 {
			return nil, fmt.Errorf("invalid tag in shape")
		}
		shapeData = shapeData[n:]

		if typ == protowire.BytesType {
			val, n := protowire.ConsumeBytes(shapeData)
			if n < 0 {
				return nil, fmt.Errorf("invalid dim bytes")
			}
			if num == shapeFieldDim {
				// dim_value is field 1, a varint; if missing the dim is symbolic/dynamic → 0
				dimVal, err := extractVarintField(val, dimFieldDimValue)
				if err != nil {
					dimVal = 0 // dynamic dim (e.g. batch size)
				}
				dims = append(dims, int64(dimVal))
			}
			shapeData = shapeData[n:]
		} else {
			n := protowire.ConsumeFieldValue(num, typ, shapeData)
			if n < 0 {
				return nil, fmt.Errorf("invalid field value in shape")
			}
			shapeData = shapeData[n:]
		}
	}

	return dims, nil
}

// extractStringField finds a string (bytes-type) field by number.
func extractStringField(data []byte, targetField protowire.Number) (string, error) {
	for len(data) > 0 {
		num, typ, n := protowire.ConsumeTag(data)
		if n < 0 {
			return "", fmt.Errorf("invalid tag")
		}
		data = data[n:]

		if typ == protowire.BytesType {
			val, n := protowire.ConsumeBytes(data)
			if n < 0 {
				return "", fmt.Errorf("invalid bytes")
			}
			if num == targetField {
				return string(val), nil
			}
			data = data[n:]
		} else {
			n := protowire.ConsumeFieldValue(num, typ, data)
			if n < 0 {
				return "", fmt.Errorf("invalid field")
			}
			data = data[n:]
		}
	}
	return "", fmt.Errorf("field %d not found", targetField)
}

// extractVarintField finds a varint field by number.
func extractVarintField(data []byte, targetField protowire.Number) (uint64, error) {
	for len(data) > 0 {
		num, typ, n := protowire.ConsumeTag(data)
		if n < 0 {
			return 0, fmt.Errorf("invalid tag")
		}
		data = data[n:]

		if typ == protowire.VarintType {
			val, n := protowire.ConsumeVarint(data)
			if n < 0 {
				return 0, fmt.Errorf("invalid varint")
			}
			if num == targetField {
				return val, nil
			}
			data = data[n:]
		} else {
			n := protowire.ConsumeFieldValue(num, typ, data)
			if n < 0 {
				return 0, fmt.Errorf("invalid field")
			}
			data = data[n:]
		}
	}
	return 0, fmt.Errorf("field %d not found", targetField)
}
