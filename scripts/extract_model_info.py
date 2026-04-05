import onnx, json, sys

model = onnx.load_model(sys.argv[1])
graph = model.graph

info = {
    "inputs": [
        {
            "name": inp.name,
            "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim],
            "dtype": inp.type.tensor_type.elem_type
        }
        for inp in graph.input
        if not any(inp.name == init.name for init in graph.initializer) 
    ],
    "outputs": [
        {
            "name": out.name,
            "shape": [d.dim_value for d in out.type.tensor_type.shape.dim],
            "dtype": out.type.tensor_type.elem_type
        }
        for out in graph.output
    ]
}

out_path = sys.argv[1].replace(".onnx", ".model_info.json")
with open(out_path, "w") as file:
    json.dump(info, file, indent=2)