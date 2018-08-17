#pragma once
// TODO: Integrate this testing (IR -> XLA) with model testing (ModelProto ->
// XLA)
// and make more formal
// What's the best testing architecture?
namespace onnx_xla {
bool almost_equal(float a, float b, float epsilon = 1e-5);
void static_relu_test();
void dynamic_relu_test();
}
