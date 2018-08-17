#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Compute Softmax
// 1) Find max of a batch
// 2) Subtract from current numbers
// 3) Exponentiate
// 4) Divide with implicit broadcasting
// TODO: Use and ENFORCE macro for checks
onnxStatus translateSoftmax(const Node& n,
                            XlaBuilder& builder,
                            ValueOpMap& valueToOp,
                            const ValueLiteralMap& valueToLiteral) {
  auto inputOp = valueToOp.at(n.inputs().at(0));
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());

  // Set axis value, defaulting to 1
  int64_t axis = 1;
  if (n.hasAttribute(kaxis)) {
    axis = n.i(kaxis);
  }

  if (axis < 0 || axis > n.inputs().at(0)->sizes().size()) {  // TODO: ENFORCE
    std::cerr << "Invalid axis attribute" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }

  // Set windowDimensions, which corresponds to a single batch
  std::vector<int64_t> windowSizes = parseOnnxInputSizes(n, 0);
  std::vector<int64> windowDimensions;
  windowDimensions.insert(windowDimensions.end(), axis, 1);
  windowDimensions.insert(windowDimensions.end(), windowSizes.begin() + axis,
                          windowSizes.end());

  // windowStrides is all 1's
  std::vector<int64> windowStrides(windowDimensions.size(), 1);

  // Compute max of each batch
  auto maxOp = builder.ReduceWindow(
      inputOp, builder.ConstantLiteral(Literal::MinValue(dataType)),
      max(dataType), windowDimensions, windowStrides, Padding::kValid);

  // Subtract max from each number (implict broadcasting)
  auto subOp = builder.Sub(inputOp, maxOp);

  // Exponentiate the result
  auto expOp = builder.Exp(subOp);

  // Sum up expOp for each batch
  auto dividendsOp = builder.ReduceWindow(
      expOp, builder.ConstantLiteral(Literal::Zero(dataType)), add(dataType),
      windowDimensions, windowStrides, Padding::kValid);

  // Build softmax by dividing
  valueToOp[n.outputs().at(0)] = builder.Div(expOp, dividendsOp);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Softmax, translateSoftmax)
}
