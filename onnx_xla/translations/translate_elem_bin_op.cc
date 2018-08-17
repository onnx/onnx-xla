#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Translates Add, Sub, Mul, and Div with broadcasting
// Implicit broadcasting if one operand has rank 0 or both have same rank
// Otherwise we explicity construct broadcastDims

#define REGISTER_ELEMENTWISE_BINARY_OP(Operation)                          \
  onnxStatus translate##Operation(const Node& n, XlaBuilder& builder,      \
                                  ValueOpMap& valueToOp,                   \
                                  const ValueLiteralMap& valueToLiteral) { \
    auto firstOp = valueToOp.at(n.inputs().at(0));                         \
    auto secondOp = valueToOp.at(n.inputs().at(1));                        \
    valueToOp[n.outputs().at(0)] = builder.Operation(                      \
        firstOp, secondOp,                                                 \
        getMultidirectionalBroadcastArg(builder, firstOp, secondOp));      \
    return ONNXIFI_STATUS_SUCCESS;                                         \
  }                                                                        \
  REGISTER_OPERATOR_TRANSLATOR(Operation, translate##Operation)

REGISTER_ELEMENTWISE_BINARY_OP(Add)
REGISTER_ELEMENTWISE_BINARY_OP(Sub)
REGISTER_ELEMENTWISE_BINARY_OP(Mul)
REGISTER_ELEMENTWISE_BINARY_OP(Div)
#undef REGISTER_ELEMENTWISE_BINARY_OP
}
