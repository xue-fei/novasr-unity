import onnx
model = onnx.load("pytorch_model_v2.onnx")
total_params = 0
for tensor in model.graph.initializer:
    total_params += tensor.raw_data.__len__() // 4  # float32
print(f"参数量: {total_params} (~{total_params*4/1024:.1f} KB)")