#include "onnx_xla/operator_registry.h"

namespace onnx_xla {

// Tranlates batch normalization in test mode
// Use BatchNormInference with input XlaOps, epsilon, and feature dimension(1)
onnxStatus translateBatchNormalization(const Node& n,
                                       XlaBuilder& builder,
                                       ValueOpMap& valueToOp,
                                       const ValueLiteralMap& valueToLiteral) {
  // If requires output of train mode, throw
  for (auto i = 1; i < n.outputs().size(); ++i) {
    if (n.outputs()[i]->uses().size() > 0) {
      throw std::runtime_error(  // TODO: ENFORCE
          "Only test mode of BatchNormalization supported");
    }
  }

  // If has attributes for train, throw
  if (n.hasAttribute(kmomentum) ||
      n.hasAttribute(Symbol("spatial"))) {  // TODO: ENFORCE
    throw std::runtime_error("Only test mode of BatchNormalization supported");
  }

  // TODO: Fetch default from ONNX Schema
  float epsilon = 1e-5;
  if (n.hasAttribute(kepsilon)) {
    epsilon = n.f(kepsilon);
  }
  valueToOp[n.outputs().at(0)] = builder.BatchNormInference(
      valueToOp.at(n.inputs().at(0)), valueToOp.at(n.inputs().at(1)),
      valueToOp.at(n.inputs().at(2)), valueToOp.at(n.inputs().at(3)),
      valueToOp.at(n.inputs().at(4)), epsilon, 1);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(BatchNormalization, translateBatchNormalization)
}
