#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Translate Concat Operator
onnxStatus translateConcat(const Node& n,
                           XlaBuilder& builder,
                           ValueOpMap& valueToOp,
                           const ValueLiteralMap& valueToLiteral) {
  if (!n.hasAttribute(kaxis)) {  // TODO: ENFORCE
    std::cerr << "Missing required axis attribute" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
  auto axis = n.i(kaxis);

  std::vector<XlaOp> inputs;
  for (auto i = 0; i < n.inputs().size(); ++i) {
    inputs.emplace_back(valueToOp.at(n.inputs()[i]));
  }

  valueToOp[n.outputs().at(0)] = builder.ConcatInDim(inputs, axis);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Concat, translateConcat)
}
