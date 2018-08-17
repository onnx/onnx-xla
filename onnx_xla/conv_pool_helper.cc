#include "onnx_xla/conv_pool_helper.h"

namespace onnx_xla {
ConvPoolHelper::ConvPoolHelper(const Node& n) {
  // Set number of spatial axes
  size_t numSpatialAxes = n.inputs().at(0)->sizes().size() - 2;
  if (numSpatialAxes <= 0) {
    throw std::runtime_error("Must have positive number of spatial axes");
  }

  // For Pool operators, prepend with dummy values for non-spatial axes
  if (n.kind() == Symbol("MaxPool") || n.kind() == Symbol("AveragePool")) {
    windowDimensions.insert(windowDimensions.end(), 2, 1);
    windowStrides.insert(windowStrides.end(), 2, 1);
    windowDilations.insert(windowDilations.end(), 2, 1);
    inputPadding.insert(inputPadding.end(), 2, std::pair<int64, int64>(0, 0));
  } else if (n.kind() != kConv) {  // TODO: ENFORCE
    throw std::runtime_error("Not a conv or pool node");
  }

  // Construct windowDimensions, windowStrides, and windowDilations
  appendVecFromSimpleAttr(kkernel_shape, n, numSpatialAxes, windowDimensions);
  appendVecFromSimpleAttr(kstrides, n, numSpatialAxes, windowStrides);
  appendVecFromSimpleAttr(kdilations, n, numSpatialAxes, windowDilations);

  // Construct inputPadding
  auto kauto_pad = Symbol("auto_pad");
  if (n.hasAttribute(kauto_pad)) {
    // Using auto_pad attribute
    auto autoPadType = n.s(kauto_pad);
    if (autoPadType == "VALID") {
      // No padding if VALID
      for (auto i = 0; i < numSpatialAxes; ++i) {
        inputPadding.emplace_back(0, 0);
      }
    } else {
      // If not VALID, then SAME_UPPER or SAME_LOWER
      // Translate input sizes into int64 vector
      const std::vector<int64_t> inputSizes = parseOnnxInputSizes(n, 0);

      // Set iterators to beginning of spatial dimensions
      auto dimensionsIt = windowDimensions.cend() - numSpatialAxes;
      auto stridesIt = windowStrides.cend() - numSpatialAxes;
      auto dilationsIt = windowDilations.cend() - numSpatialAxes;
      auto sizesIt = inputSizes.cend() - numSpatialAxes;

      // Compute axisPadding using formula in docs for SAME_UPPER and SAME_LOWER
      std::vector<int64_t> axisPadding;
      for (auto i = 0; i < numSpatialAxes; ++i) {
        auto outputShape = (*sizesIt + *stridesIt - 1) / *stridesIt;
        axisPadding.emplace_back((outputShape - 1) * *stridesIt +
                                 *dimensionsIt * *dilationsIt - *sizesIt);
        ++dimensionsIt;
        ++stridesIt;
        ++dilationsIt;
        ++sizesIt;
      }

      if (autoPadType == "SAME_UPPER") {
        // Emplace (floor, ceiling) of axisPadding/2
        for (auto i = 0; i < numSpatialAxes; ++i) {
          inputPadding.emplace_back(axisPadding.at(i) / 2,
                                    (axisPadding.at(i) + 1) / 2);
        }
      } else if (autoPadType == "SAME_LOWER") {
        // Emplace (ceiling, floor) of axisPadding/2
        for (auto i = 0; i < numSpatialAxes; ++i) {
          inputPadding.emplace_back((axisPadding.at(i) + 1) / 2,
                                    axisPadding.at(i) / 2);
        }
      } else {  // TODO: ENFORCE
        throw std::runtime_error("Invalid auto_pad attribute");
      }
    }
  } else {
    // Not using auto_pads attribute
    if (!n.hasAttribute(kpads)) {
      // If not pads attribute, default to no padding
      for (auto i = 0; i < numSpatialAxes; ++i) {
        inputPadding.emplace_back(0, 0);
      }
    } else {
      // If pads attribute, fill in padding with pairs from pads
      auto pads = n.is(kpads);
      for (auto i = 0; i < numSpatialAxes; ++i) {
        inputPadding.emplace_back(pads.at(i), pads.at(i + numSpatialAxes));
      }
    }
  }
}

void ConvPoolHelper::appendVecFromSimpleAttr(const Symbol& attr,
                                             const Node& n,
                                             size_t numSpatialAxes,
                                             std::vector<int64>& vec) {
  if (!n.hasAttribute(attr)) {
    if (attr == kkernel_shape && n.kind() == kConv) {
      const std::vector<int64_t> kernelDims = parseOnnxInputSizes(n, 1);
      if (kernelDims.size() != numSpatialAxes) {
        throw std::runtime_error("Valid kernel shape could not be inferred");
      }
      vec.insert(vec.end(), kernelDims.begin(), kernelDims.end());
    } else {
      vec.insert(vec.end(), numSpatialAxes, 1);
    }
  } else {
    const auto& attrShape = n.is(attr);
    if (attrShape.size() != numSpatialAxes) {  // TODO: ENFORCE
      throw std::runtime_error("Mismatching attribute shape");
    }
    vec.reserve(attrShape.size());
    vec.insert(vec.end(), attrShape.begin(), attrShape.end());
  }
}

const std::vector<int64>& ConvPoolHelper::getWindowDimensions() const {
  return windowDimensions;
}

const std::vector<int64>& ConvPoolHelper::getWindowStrides() const {
  return windowStrides;
}

const std::vector<int64>& ConvPoolHelper::getWindowDilations() const {
  return windowDilations;
}

const std::vector<std::pair<int64, int64>>& ConvPoolHelper::getInputPadding()
    const {
  return inputPadding;
}
}
