#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// TODO: Handle Undefined properly
onnxStatus translateUndefined(const Node& n,
                              XlaBuilder& builder,
                              ValueOpMap& valueToOp,
                              const ValueLiteralMap& valueToLiteral) {
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Undefined, translateUndefined)
}
