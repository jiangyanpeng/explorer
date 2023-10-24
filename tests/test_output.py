import onnx
import sys
from onnx import shape_inference


a = sys.argv[1]
original_model = onnx.load(a)
inferred_model = shape_inference.infer_shapes(original_model)
print(inferred_model.graph.value_info)