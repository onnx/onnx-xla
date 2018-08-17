#include "onnx_xla/utils.h"

namespace onnx_xla {
xla::PrimitiveType onnxToPrimitive(
    const ONNX_NAMESPACE::TensorProto_DataType& data_type) {
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      return xla::F32;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
      return xla::C64;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
      return xla::F16;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL: {
      return xla::PRED;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      return xla::S8;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      return xla::S16;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      return xla::S32;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      return xla::U8;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      return xla::U16;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      return xla::S64;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {
      return xla::U32;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
      return xla::U64;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
      return xla::F64;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED: {
      throw std::runtime_error("Not supported");
    }
  }
}

XlaComputation add(PrimitiveType dataType) {
  XlaBuilder builder("add");
  auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
  auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
  builder.Add(y, x);
  return builder.Build().ConsumeValueOrDie();
}

XlaComputation max(PrimitiveType dataType) {
  XlaBuilder builder("max");
  auto y = builder.Parameter(0, ShapeUtil::MakeShape(dataType, {}), "y");
  auto x = builder.Parameter(1, ShapeUtil::MakeShape(dataType, {}), "x");
  builder.Max(y, x);
  return builder.Build().ConsumeValueOrDie();
}

std::vector<int64_t> parseOnnxInputSizes(const Node& n, size_t inputIndex) {
  if (!n.inputs().at(inputIndex)->has_sizes()) {  // TODO: Enforce
    throw std::runtime_error("Missing shape");
  }
  std::vector<int64_t> shapeInts;
  const auto& shapeDims = n.inputs().at(inputIndex)->sizes();
  for (const auto& dimension : shapeDims) {
    if (!dimension.is_int) {  // TODO: Enforce
      throw std::runtime_error("Invalid dimension");
    }
    shapeInts.emplace_back(dimension.dim);
  }
  return shapeInts;
}

std::vector<int64> getMultidirectionalBroadcastArg(const XlaBuilder& builder,
                                                   const XlaOp& firstOp,
                                                   const XlaOp& secondOp) {
  auto firstNDim = ShapeUtil::Rank(builder.GetShape(firstOp).ValueOrDie());
  auto secondNDim = ShapeUtil::Rank(builder.GetShape(secondOp).ValueOrDie());
  std::vector<int64> broadcastDims;
  if (firstNDim != secondNDim || firstNDim != 0 || secondNDim != 0) {
    auto minDim = std::min(firstNDim, secondNDim);
    auto maxDim = std::max(firstNDim, secondNDim);
    for (auto j = 0; j < minDim; ++j) {
      broadcastDims.push_back(j + maxDim - minDim);
    }
  }
  return broadcastDims;
}
}
