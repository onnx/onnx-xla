#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Compute relu by taking max between the input and constant zero literal
onnxStatus translateRelu(const Node& n,
                         XlaBuilder& builder,
                         ValueOpMap& valueToOp,
                         const ValueLiteralMap& valueToLiteral) {
  auto input = valueToOp[n.inputs().at(0)];
  auto zero = ::tensorflow::FloatLiteral(
      &builder, onnxToPrimitive(n.inputs().at(0)->elemType()), 0);
  auto maximum = builder.Max(input, zero);
  valueToOp[n.outputs().at(0)] = maximum;
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Relu, translateRelu)
}
