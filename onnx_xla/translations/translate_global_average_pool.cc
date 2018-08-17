#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Compute GlobalAveragePool
// 1. Use Reduce window with add computation
// 2. Divide by size of window
onnxStatus translateGlobalAveragePool(const Node& n,
                                      XlaBuilder& builder,
                                      ValueOpMap& valueToOp,
                                      const ValueLiteralMap& valueToLiteral) {
  // Set dataType
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());

  // Set 1 window per batch and channel
  std::vector<int64_t> windowSizes = parseOnnxInputSizes(n, 0);
  std::vector<int64> windowDimensions;
  windowDimensions.insert(windowDimensions.end(), 2, 1);
  windowDimensions.insert(windowDimensions.end(), windowSizes.begin() + 2,
                          windowSizes.end());

  // Set strides
  std::vector<int64> windowStrides(windowDimensions.size(), 1);

  // Execute pooling and division
  auto PoolOp = builder.ReduceWindow(
      valueToOp.at(n.inputs().at(0)),
      builder.ConstantLiteral(Literal::Zero(dataType)), add(dataType),
      windowDimensions, windowStrides, Padding::kValid);

  auto numWindowElements =
      std::accumulate(windowDimensions.begin(), windowDimensions.end(), 1L,
                      std::multiplies<int64>());
  auto numOp =
      ::tensorflow::FloatLiteral(&builder, dataType, numWindowElements);
  valueToOp[n.outputs().at(0)] = builder.Div(PoolOp, numOp);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(GlobalAveragePool, translateGlobalAveragePool)
}
