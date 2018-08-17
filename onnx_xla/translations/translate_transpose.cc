#include "onnx_xla/operator_registry.h"

namespace onnx_xla {
// Translate Transpose
onnxStatus translateTranspose(const Node& n,
                              XlaBuilder& builder,
                              ValueOpMap& valueToOp,
                              const ValueLiteralMap& valueToLiteral) {
  // Build permutation
  std::vector<int64> permutation;
  if (n.hasAttribute(kperm)) {
    // Use attribute values as permutation
    permutation.insert(permutation.begin(), n.is(kperm).cbegin(),
                       n.is(kperm).cend());
  } else {
    // If attribute not provided, reverse as default
    int64 numDims = n.inputs().at(0)->sizes().size();
    if (numDims == 0) {  // TODO: ENFORCE
      std::cerr << "Missing input shape size" << std::endl;
      return ONNXIFI_STATUS_INVALID_MODEL;
    }
    while (--numDims >= 0) {
      permutation.emplace_back(numDims);
    }
  }
  // Execute transpose of input
  valueToOp[n.outputs().at(0)] =
      builder.Transpose(valueToOp.at(n.inputs().at(0)), permutation);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Transpose, translateTranspose)
}
