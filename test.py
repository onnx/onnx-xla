from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnxifi_backend import OnnxifiBackend, OnnxifiBackendRep
import onnx
from onnx import ModelProto, NodeProto
from onnx import numpy_helper
import numpy as np

# This is for experimental purposes
backend = OnnxifiBackend()

print("Available devices")
print(backend.get_devices_info())
assert(backend.supports_device("CPU"))
assert(not backend.supports_device("GPU"))


# Temporary Relu test case
node = onnx.helper.make_node(
      'Relu', ['x'], ['y'], name='relu_node')
graph = onnx.helper.make_graph(
    nodes=[node],
    name='SingleRelu',
    inputs=[onnx.helper.make_tensor_value_info(
        'x', onnx.TensorProto.FLOAT, [1, 2])],
    outputs=[onnx.helper.make_tensor_value_info(
        'y', onnx.TensorProto.FLOAT, [1, 2])])
model = onnx.helper.make_model(graph, producer_name='backend-test')
onnx.checker.check_model(model)

assert(backend.is_compatible(model, device='CPU'))
backendrep = backend.prepare(model, device='CPU')

x = np.random.randn(1, 2).astype(np.float32)
y = np.maximum(x, 0)

outputs = backendrep.run([x])
expected_outputs = [y]
np.testing.assert_equal(expected_outputs, outputs)

#Running graph again
x = np.random.randn(1, 2).astype(np.float32)
y = np.maximum(x, 0)

outputs = backendrep.run([x])
expected_outputs = [y]
np.testing.assert_equal(expected_outputs, outputs)

# test reshape
# TODO: Make this unit case as node tests have dynamic shape
original_shape = [2, 3, 4]
test_cases = {
    'reordered_dims': ([4, 2, 3], [4, 2, 3]),
    'reduced_dims': ([3, 8], [3, 8]),
    'extended_dims': ([3, 2, 2, 2], [3, 2, 2, 2]),
    'one_dim': ([24], [24]),
    'negative_dim': ([6, -1, 2], [6, 2, 2]),
    'zero_dim': ([4, 0, 2], [4, 3, 2]),
}
data = np.random.random_sample(original_shape).astype(np.float32)

for test_name, shape in test_cases.items():
    target_shape_input = shape[0]
    target_shape = shape[1]
    reshaped = np.reshape(data, target_shape)

    node = onnx.helper.make_node(
        'Reshape',
        inputs=['data', 'shape'],
        outputs=['reshaped'],
    )
    graph = onnx.helper.make_graph(
        nodes=[node],
        name='SingleReshape',
        inputs=[onnx.helper.make_tensor_value_info(
            'data', onnx.TensorProto.FLOAT, original_shape),
                onnx.helper.make_tensor_value_info(
            'shape', onnx.TensorProto.INT64, [len(target_shape_input)])],
        outputs=[onnx.helper.make_tensor_value_info(
            'reshaped', onnx.TensorProto.FLOAT, target_shape)],
        initializer=[onnx.helper.make_tensor(
            'shape', onnx.TensorProto.INT64, [len(target_shape_input)], target_shape_input)])

    model = onnx.helper.make_model(graph, producer_name='backend-test')
    onnx.checker.check_model(model)

    assert(backend.is_compatible(model, device='CPU'))
    backendrep = backend.prepare(model, device='CPU')
    
    outputs = backendrep.run([data])
    expected_outputs = [reshaped]
    np.testing.assert_equal(expected_outputs, outputs)

