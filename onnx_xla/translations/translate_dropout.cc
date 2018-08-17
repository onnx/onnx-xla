#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Translate Dropout for test mode
// Output is the same as the input
// Throw if mask exists and has use
onnxStatus translateDropout(const Node& n,
                            XlaBuilder& builder,
                            ValueOpMap& valueToOp,
                            const ValueLiteralMap& valueToLiteral) {
  if (n.outputs().size() > 1 &&
      n.outputs()[1]->uses().size() > 0) {  // TODO:ENFORCE
    throw std::runtime_error("Dropout only supported in test mode");
  }
  valueToOp[n.outputs().at(0)] = valueToOp.at(n.inputs().at(0));
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Dropout, translateDropout)
}
