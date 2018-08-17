#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Translate LRN
// 1. Square input
// 2. Use reduce to sum squares
// 3. Do arithmetic to finish
onnxStatus translateLRN(const Node& n,
                        XlaBuilder& builder,
                        ValueOpMap& valueToOp,
                        const ValueLiteralMap& valueToLiteral) {
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());
  // Read in attributes and make them XlaOp
  // TODO: Read default from schema
  float alpha = 1e-4f;
  if (n.hasAttribute(kalpha)) {
    alpha = n.f(kalpha);
  }
  auto alphaOp = ::tensorflow::FloatLiteral(&builder, dataType, alpha);

  float beta = 0.75f;
  if (n.hasAttribute(kbeta)) {
    beta = n.f(kbeta);
  }
  auto betaOp = ::tensorflow::FloatLiteral(&builder, dataType, beta);

  float bias = 1.0f;
  auto kbias = Symbol("bias");
  if (n.hasAttribute(kbias)) {
    bias = n.f(kbias);
  }
  auto biasOp = ::tensorflow::FloatLiteral(&builder, dataType, bias);

  if (!n.hasAttribute(ksize)) {  // TODO: Enforce
    std::cerr << "Missing required size attribute" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
  auto size = n.i(ksize);
  auto sizeOp = ::tensorflow::FloatLiteral(&builder, dataType, size);

  // Square input
  auto inputOp = valueToOp.at(n.inputs().at(0));
  auto squaredOp = builder.Mul(inputOp, inputOp);

  // Pool with add, making the required attributes
  // Note: kSame pads 1 more in the higher values of a dimension when the
  // padding is odd (which corresponds to ONNX LRN operator definition)

  std::vector<int64> windowDimensions(n.inputs().at(0)->sizes().size(), 1);
  windowDimensions.at(1) = size;
  std::vector<int64> windowStrides(windowDimensions.size(), 1);

  auto sumSquaresOp = builder.ReduceWindow(
      squaredOp, builder.ConstantLiteral(Literal::Zero(dataType)),
      add(dataType), windowDimensions, windowStrides, Padding::kSame);

  // Do final arithmetic
  valueToOp[n.outputs().at(0)] = builder.Div(
      inputOp,
      builder.Pow(builder.Add(biasOp, builder.Mul(builder.Div(alphaOp, sizeOp),
                                                  sumSquaresOp)),
                  betaOp));
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(LRN, translateLRN)
}
