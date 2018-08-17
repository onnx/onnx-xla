#include "onnx_xla/operator_registry.h"

namespace onnx_xla {

// TODO: enforce macros
// 1) Transpose matrices, if necessary
// 2) Multiply A and B
// 3) Scale by alpha
// 4) Scale C by beta
// 5) Add with unidirectional broadcasting
onnxStatus translateGemm(const Node& n,
                         XlaBuilder& builder,
                         ValueOpMap& valueToOp,
                         const ValueLiteralMap& valueToLiteral) {
  // Type checks
  auto onnxType = n.inputs().at(0)->elemType();
  if (onnxType != n.inputs().at(1)->elemType() ||
      onnxType != n.inputs().at(2)->elemType()) {  // TODO: enforce
    std::cerr << "Data types of inputs do not match" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
  auto dataType = onnxToPrimitive(onnxType);

  // Set transpose values and do shape checks
  auto AOp = valueToOp.at(n.inputs().at(0));
  auto BOp = valueToOp.at(n.inputs().at(1));
  ::xla::DotDimensionNumbers dnums;
  auto shapeA = builder.GetShape(AOp).ValueOrDie();
  auto shapeB = builder.GetShape(BOp).ValueOrDie();
  if (ShapeUtil::Rank(shapeA) != 2 ||
      ShapeUtil::Rank(shapeB) != 2) {  // TODO: enforce
    std::cerr << "Operands to multiply must be 2 dimensional" << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
  auto dimA = ShapeUtil::GetDimension(shapeA, 1);
  auto dimB = ShapeUtil::GetDimension(shapeB, 0);

  if (n.hasAttribute(ktransA) && n.i(ktransA) != 0) {
    dimA = ShapeUtil::GetDimension(shapeA, 0);
    dnums.add_lhs_contracting_dimensions(0);
  } else {
    dnums.add_lhs_contracting_dimensions(1);
  }

  if (n.hasAttribute(ktransB) && n.i(ktransB) != 0) {
    dimB = ShapeUtil::GetDimension(shapeB, 1);
    dnums.add_rhs_contracting_dimensions(1);
  } else {
    dnums.add_rhs_contracting_dimensions(0);
  }

  if (dimA != dimB) {
    std::cerr << "Incompatible dimensions for matrix multiplication"
              << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  }

  // Compute alpha * A * B
  auto ABOp = builder.DotGeneral(AOp, BOp, dnums);
  if (n.hasAttribute(kalpha)) {
    auto alphaOp = ::tensorflow::FloatLiteral(&builder, dataType, n.f(kalpha));
    ABOp = builder.Mul(ABOp, alphaOp);
  }

  // Compute beta * C
  auto COp = valueToOp.at(n.inputs().at(2));
  if (n.hasAttribute(kbeta)) {
    auto betaOp = ::tensorflow::FloatLiteral(&builder, dataType, n.f(kbeta));
    COp = builder.Mul(COp, betaOp);
  }

  // Add together, potentially with boadcasting
  // TODO: Verify shapes unidirectionally broadcastable
  // If correct, XLA will implicitly broadcast scalar and same rank arrays
  std::vector<int64> broadcastDims;
  auto rankC = ShapeUtil::Rank(builder.GetShape(COp).ValueOrDie());
  if (rankC == 1) {
    broadcastDims.emplace_back(1);
  }
  valueToOp[n.outputs().at(0)] = builder.Add(ABOp, COp, broadcastDims);
  return ONNXIFI_STATUS_SUCCESS;
}
REGISTER_OPERATOR_TRANSLATOR(Gemm, translateGemm)
}
