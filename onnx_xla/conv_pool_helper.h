#pragma once

#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Utility struct to help translate Conv/MaxPool/AveragePool operators
struct ConvPoolHelper {
 public:
  // Given a node of kind Conv, AveragePool, or MaxPool will set the
  // window dimensions, strides, and dilations as well as the input padding
  // in the format compatible with XlaBuilder
  ConvPoolHelper(const Node& n);

  // Getter functions for relevant attributes
  const std::vector<int64>& getWindowDimensions() const;

  const std::vector<int64>& getWindowStrides() const;

  const std::vector<int64>& getWindowDilations() const;

  const std::vector<std::pair<int64, int64>>& getInputPadding() const;

 private:
  // Utility function to translate "simple" ONNX attributes (int vectors) to
  // format required by XlaBuilder
  // If n has attribute attr: Append to vec the values from attr (must have size
  // numSpatialAxes)
  // Else: Fill with numSpatialAxes 1's (or if attr is kkernel_shape and n is of
  // kind kConv, infer from inputs)
  static void appendVecFromSimpleAttr(const Symbol& attr,
                                      const Node& n,
                                      size_t numSpatialAxes,
                                      std::vector<int64>& vec);

  // State that's required to translate Pool/Conv operators
  std::vector<int64> windowDimensions;
  std::vector<int64> windowStrides;
  std::vector<int64> windowDilations;
  std::vector<std::pair<int64, int64>> inputPadding;
};
}
