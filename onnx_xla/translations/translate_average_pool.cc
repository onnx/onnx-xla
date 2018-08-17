#include "onnx_xla/operator_registry.h"
#include "onnx_xla/conv_pool_helper.h"

namespace onnx_xla {

onnxStatus translateAveragePool(const Node& n,
                                XlaBuilder& builder,
                                ValueOpMap& valueToOp,
                                const ValueLiteralMap& valueToLiteral) {
  // Set dataType
  auto dataType = onnxToPrimitive(n.inputs().at(0)->elemType());

  // Create ConvPoolHelper object (constructs attributes formatted for
  // XlaBuilder)
  ConvPoolHelper helper(n);

  // Enque a sum Xla operation
  XlaOp sumOp = builder.ReduceWindowWithGeneralPadding(
      valueToOp.at(n.inputs().at(0)),
      builder.ConstantLiteral(Literal::Zero(dataType)), add(dataType),
      helper.getWindowDimensions(), helper.getWindowStrides(),
      helper.getInputPadding());

  // Build up divisor
  XlaOp divisorOp;
  auto kcount_include_pad = Symbol("count_include_pad");
  if (!n.hasAttribute(kcount_include_pad) || n.i(kcount_include_pad) == 0) {
    // To not include pads, send computation to XLA to calculate a literal
    // of the windowSizes
    XlaBuilder windowBuilder("windowBuilder");
    std::vector<int64_t> inputSizes = parseOnnxInputSizes(n, 0);
    std::vector<int64> onesSizes(inputSizes.begin(), inputSizes.end());
    auto ones = windowBuilder.Broadcast(
        ::tensorflow::FloatLiteral(&windowBuilder, dataType, 1.), onesSizes);
    auto windowSizes = windowBuilder.ReduceWindowWithGeneralPadding(
        ones, windowBuilder.ConstantLiteral(Literal::Zero(dataType)),
        add(dataType), helper.getWindowDimensions(), helper.getWindowStrides(),
        helper.getInputPadding());
    auto windowComputationStatus = windowBuilder.Build();
    if (!windowComputationStatus.ok()) {
      throw std::runtime_error("The computation to find window sizes failed.");
    }
    auto windowComputation = windowComputationStatus.ConsumeValueOrDie();
    auto result = xla::ExecuteComputation(windowComputation, {});
    divisorOp = builder.ConstantLiteral(*result);
  } else {
    // Including pads, we just have one scalar XlaOp with the windowSize
    auto windowSize = std::accumulate(helper.getWindowDimensions().cbegin(),
                                      helper.getWindowDimensions().cend(), 1L,
                                      std::multiplies<int64>());
    auto divisorOp = ::tensorflow::FloatLiteral(&builder, dataType, windowSize);
  }
  // Do division, with implicit broadcasting
  valueToOp[n.outputs().at(0)] = builder.Div(sumOp, divisorOp);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(AveragePool, translateAveragePool)
}
