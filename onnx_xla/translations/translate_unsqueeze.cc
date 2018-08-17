#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Translate Unsqueeze by using XLA's Reshape
onnxStatus translateUnsqueeze(const Node& n,
                              XlaBuilder& builder,
                              ValueOpMap& valueToOp,
                              const ValueLiteralMap& valueToLiteral) {
  // Set origShape and axes
  std::vector<int64_t> origShape = parseOnnxInputSizes(n, 0);

  if (!n.hasAttribute(kaxes)) {  // TODO ENFORCE
    std::cerr << "Missing Required Attribute" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
  const auto& axes = n.is(kaxes);

  // Set 1's then rest of values to  make newShape
  std::vector<int64> newShape(origShape.size() + axes.size());
  for (auto axis : axes) {
    newShape.at(axis) = 1;
  }
  for (auto i = 0, j = 0; i < newShape.size(); ++i) {
    if (newShape[i] != 1) {
      newShape[i] = origShape[j++];
    }
  }

  // Enqueue operation
  valueToOp[n.outputs().at(0)] =
      builder.Reshape(valueToOp.at(n.inputs().at(0)), newShape);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Unsqueeze, translateUnsqueeze)
}
