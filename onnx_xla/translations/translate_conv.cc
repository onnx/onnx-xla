#include "onnx_xla/operator_registry.h"
#include "onnx_xla/conv_pool_helper.h"

namespace onnx_xla {

onnxStatus translateConv(const Node& n,
                         XlaBuilder& builder,
                         ValueOpMap& valueToOp,
                         const ValueLiteralMap& valueToLiteral) {
  // Create ConvPoolHelper object (constructs attributes formatted for
  // XlaBuilder)
  ConvPoolHelper helper(n);
  // Set inputOp and kernelOp
  auto inputOp = valueToOp.at(n.inputs().at(0));
  auto windowOp = valueToOp.at(n.inputs().at(1));

  // Build result of convolution
  XlaOp convOp;
  if (!n.hasAttribute(kgroup) || n.i(kgroup) == 1) {
    // Enque corresponding Xla operation
    convOp = builder.ConvGeneralDilated(
        inputOp, windowOp, helper.getWindowStrides(), helper.getInputPadding(),
        {}, helper.getWindowDilations(),
        XlaBuilder::CreateDefaultConvDimensionNumbers(
            helper.getWindowStrides().size()));
  } else {
    const int numGroups = n.i(kgroup);
    std::vector<int64_t> inputDims = parseOnnxInputSizes(n, 0);
    std::vector<int64_t> windowDims = parseOnnxInputSizes(n, 1);
    if (inputDims.at(1) % numGroups != 0 || windowDims.at(0) % numGroups != 0) {
      throw std::runtime_error(
          "Input and kernel channel numbers should be divisible by group "
          "number");
    }
    auto inputChannelsPerGroup = inputDims[1] / numGroups;
    auto outputChannelsPerGroup = windowDims[0] / numGroups;
    auto inputStartIndex = 0;
    auto windowStartIndex = 0;

    std::vector<XlaOp> convOps;
    for (auto i = 0; i < numGroups; ++i) {
      auto inputSliceOp =
          builder.SliceInDim(inputOp, inputStartIndex,
                             inputStartIndex + inputChannelsPerGroup, 1, 1);
      auto windowSliceOp =
          builder.SliceInDim(windowOp, windowStartIndex,
                             windowStartIndex + outputChannelsPerGroup, 1, 0);

      convOps.push_back(builder.ConvGeneralDilated(
          inputSliceOp, windowSliceOp, helper.getWindowStrides(),
          helper.getInputPadding(), {}, helper.getWindowDilations(),
          XlaBuilder::CreateDefaultConvDimensionNumbers(
              helper.getWindowStrides().size())));

      inputStartIndex += inputChannelsPerGroup;
      windowStartIndex += outputChannelsPerGroup;
    }
    convOp = builder.ConcatInDim(convOps, 1);
  }

  // Add optional bias and finish
  if (n.inputs().size() == 3 && n.inputs().at(2)->uniqueName() != "") {
    XlaOp biasOp = valueToOp.at(n.inputs().at(2));
    convOp = builder.Add(convOp, biasOp, {1});
  }
  valueToOp[n.outputs().at(0)] = convOp;

  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Conv, translateConv)
}
