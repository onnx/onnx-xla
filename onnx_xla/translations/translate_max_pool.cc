#include "onnx_xla/operator_registry.h"
#include "onnx_xla/conv_pool_helper.h"

namespace onnx_xla {
// Translates Max Pool (no indices support yet)
// TODO: Support for indices output
// TODO: storage_order attribute
onnxStatus translateMaxPool(const Node& n,
                            XlaBuilder& builder,
                            ValueOpMap& valueToOp,
                            const ValueLiteralMap& valueToLiteral) {
  // Check if indices output required
  if (n.outputs().size() > 1 &&
      n.outputs().at(1)->uses().size() > 0) {  // TODO:Enforce
    throw std::runtime_error("MaxPool with indices output not yet supported");
  }

  auto kstorage_order = Symbol("storage_order");
  if (n.hasAttribute(kstorage_order) &&
      n.i(kstorage_order) != 0) {  // TODO: Enforce
    throw std::runtime_error(
        "MaxPool with column-major storage order not yet supported");
  }

  // Set dataType for computations
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());

  // Create ConvPoolHelper object (constructs attributes formatted for
  // XlaBuilder)
  ConvPoolHelper helper(n);

  // Enque corresponding Xla operation
  valueToOp[n.outputs().at(0)] = builder.ReduceWindowWithGeneralPadding(
      valueToOp.at(n.inputs().at(0)),
      builder.ConstantLiteral(Literal::MinValue(dataType)), max(dataType),
      helper.getWindowDimensions(), helper.getWindowStrides(),
      helper.getInputPadding());
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(MaxPool, translateMaxPool)
}
