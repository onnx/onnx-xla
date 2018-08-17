#include "python_onnxifi/data_conversion.h"

#include <utility>
#include <unordered_set>
#include <algorithm>
#include <complex>
#include <functional>
namespace py = pybind11;

#define DISPATCH_OVER_NUMERIC_DATA_TYPE(data_type, op_template, ...)        \
  switch (data_type) {                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {                      \
      op_template<float, float>(__VA_ARGS__);                               \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {                  \
      op_template<std::complex<float>, std::complex<float>>(__VA_ARGS__);   \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {                    \
      /*TODO:op_template<int32_t, half>(__VA_ARGS__);*/                     \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL: {                       \
      op_template<int32_t, bool>(__VA_ARGS__);                              \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {                       \
      op_template<int32_t, int8_t>(__VA_ARGS__);                            \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {                      \
      op_template<int32_t, int16_t>(__VA_ARGS__);                           \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {                      \
      op_template<int32_t, int32_t>(__VA_ARGS__);                           \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {                      \
      op_template<int32_t, uint8_t>(__VA_ARGS__);                           \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {                     \
      op_template<int32_t, uint16_t>(__VA_ARGS__);                          \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {                      \
      op_template<int64_t, int64_t>(__VA_ARGS__);                           \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {                     \
      op_template<uint64_t, uint32_t>(__VA_ARGS__);                         \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {                     \
      op_template<uint64_t, uint64_t>(__VA_ARGS__);                         \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {                     \
      op_template<double, double>(__VA_ARGS__);                             \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {                 \
      op_template<std::complex<double>, std::complex<double>>(__VA_ARGS__); \
      break;                                                                \
    }                                                                       \
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:                       \
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED: {                  \
      throw std::runtime_error("Dispatch received non-numeric data type");  \
    }                                                                       \
  }

DescriptorData::DescriptorData(DescriptorData&& d) noexcept {
  name = std::move(d.name);
  buffer = std::move(d.buffer);
  shape = std::move(d.shape);
  descriptor.tag = d.descriptor.tag;
  descriptor.name = name.c_str();
  descriptor.dimensions = shape.size();
  descriptor.shape = shape.data();
  descriptor.buffer = reinterpret_cast<onnxPointer>(buffer.data());
  descriptor.memoryType = d.descriptor.memoryType;
  descriptor.dataType = d.descriptor.dataType;
}

DescriptorData::DescriptorData(const DescriptorData& d) {
  name = d.name;
  buffer = d.buffer;
  shape = d.shape;
  descriptor.tag = d.descriptor.tag;
  descriptor.name = name.c_str();
  descriptor.dimensions = shape.size();
  descriptor.shape = shape.data();
  descriptor.buffer = reinterpret_cast<onnxPointer>(buffer.data());
  descriptor.memoryType = d.descriptor.memoryType;
  descriptor.dataType = d.descriptor.dataType;
}

DescriptorData::DescriptorData(const ValueInfoProto& vip) {
  descriptor.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  name = vip.name();
  descriptor.name = name.c_str();
  if (!vip.type().tensor_type().has_elem_type()) {  // TODO: ENFORCE_EQ
    throw std::runtime_error("Non-static ModelProto: Data type not found");
  }
  descriptor.dataType = vip.type().tensor_type().elem_type();  // TODO
                                                               // ENFORCE_EQ
  descriptor.dimensions =
      vip.type().tensor_type().shape().dim_size();  // TODO ENFORCE_EQ
  for (auto i = 0; i < descriptor.dimensions; ++i) {
    const auto& dim = vip.type().tensor_type().shape().dim(i);
    if (!dim.has_dim_value()) {  // TODO: ENFORCE_EQ
      throw std::runtime_error(
          "Non-static ModelProto: Shape dimension not found");
    }
    shape.emplace_back(dim.dim_value());
  }
  descriptor.shape = shape.data();
  descriptor.memoryType =
      ONNXIFI_MEMORY_TYPE_CPU;  // TODO: Expand memory types?
  uint64_t buffer_size;
  DISPATCH_OVER_NUMERIC_DATA_TYPE(descriptor.dataType, getBufferSize,
                                  buffer_size, descriptor.shape,
                                  descriptor.dimensions)
  buffer.resize(buffer_size);
  descriptor.buffer = reinterpret_cast<onnxPointer>(buffer.data());
}

template <typename onnx_type, typename unused>
void DescriptorData::getBufferSize(uint64_t& buffer_size,
                                   const uint64_t* shape,
                                   uint32_t dimensions) {
  auto numElements = std::accumulate(shape, shape + dimensions, 1UL,
                                     std::multiplies<uint64_t>());
  buffer_size = sizeof(onnx_type) * numElements;
}

// TODO: Addweight descriptors
DataConversion::DataConversion(const std::string& serializedModel) {
  if (!model_.ParseFromString(serializedModel)) {
    throw std::runtime_error("Failed to parse model proto");
  }
  ::ONNX_NAMESPACE::shape_inference::InferShapes(model_);
  const auto& initializers = model_.graph().initializer();
  const auto& inputs = model_.graph().input();
  const auto& outputs = model_.graph().output();
  std::unordered_set<std::string> initializerNames;
  for (const auto& t : initializers) {
    initializerNames.insert(t.name());
  }
  for (const auto& vip : inputs) {
    std::string name = vip.name();
    if (initializerNames.find(name) == initializerNames.end()) {
      input_descriptors_data_.emplace_back(vip);
    }
  }
  for (const auto& vip : outputs) {
    output_descriptors_data_.emplace_back(vip);
  }
}

template <typename onnx_type, typename py_type>
void DataConversion::fillDescriptorDataImpl(
    DescriptorData& dd,
    const py::array& numpyArray,
    ONNX_NAMESPACE::TensorProto_DataType dataType) {
  // TODO: ENFORCE_EQ
  if (dd.descriptor.name != dd.name.c_str())
    throw std::runtime_error("Wrong input data type");
  if (dd.descriptor.shape != dd.shape.data())
    throw std::runtime_error("Wrong input data type");
  if (reinterpret_cast<char*>(dd.descriptor.buffer) != dd.buffer.data())
    throw std::runtime_error("Wrong input data type");
  if (dd.descriptor.dataType != dataType)
    throw std::runtime_error("Wrong input data type");
  if (dd.descriptor.dimensions != numpyArray.ndim())
    throw std::runtime_error("Wrong input dimension");
  if (!std::equal(numpyArray.shape(), numpyArray.shape() + numpyArray.ndim(),
                  dd.shape.begin()))
    throw std::runtime_error("Wrong input shape");
  onnx_type* buffer = reinterpret_cast<onnx_type*>(dd.buffer.data());
  const py_type* data = reinterpret_cast<const py_type*>(numpyArray.data());
  std::copy(data, data + numpyArray.size(), buffer);
  dd.descriptor.buffer = reinterpret_cast<onnxPointer>(buffer);
}

void DataConversion::fillDescriptorDataVector(
    const py::list& numpyArrayList,
    std::vector<DescriptorData>& descriptorsData) {
  if (descriptorsData.size() != numpyArrayList.size()) {  // TODO: ENFORCE_EQ
    throw std::runtime_error("Incompatible vector sizes");
  }
  auto ddIterator = descriptorsData.begin();
  auto npIterator = numpyArrayList.begin();
  while (npIterator != numpyArrayList.end()) {
    auto& dd = *ddIterator;
    const auto numpyArray = py::reinterpret_borrow<py::array>(*npIterator);
    const auto dtype = numpyArray.dtype();
    if (dtype.is(py::dtype::of<bool>())) {
      fillDescriptorDataImpl<int32_t, bool>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_BOOL);
    } else if (dtype.is(py::dtype::of<int8_t>())) {
      fillDescriptorDataImpl<int32_t, int8_t>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_INT8);
    } else if (dtype.is(py::dtype::of<int16_t>())) {
      fillDescriptorDataImpl<int32_t, int16_t>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_INT16);
    } else if (dtype.is(py::dtype::of<int32_t>())) {
      fillDescriptorDataImpl<int32_t, int32_t>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_INT32);
    } else if (dtype.is(py::dtype::of<int64_t>())) {
      fillDescriptorDataImpl<int64_t, int64_t>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_INT64);
    } else if (dtype.is(py::dtype::of<uint8_t>())) {
      fillDescriptorDataImpl<int32_t, uint8_t>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    } else if (dtype.is(py::dtype::of<uint16_t>())) {
      fillDescriptorDataImpl<int32_t, uint16_t>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_UINT16);
    } else if (dtype.is(py::dtype::of<uint32_t>())) {
      fillDescriptorDataImpl<uint64_t, uint32_t>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_UINT32);
    } else if (dtype.is(py::dtype::of<uint64_t>())) {
      fillDescriptorDataImpl<uint64_t, uint64_t>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_UINT64);
    } /*else if (dtype.is(py::dtype::of<half>())) {
    TODO: fillDescriptorDataImpl<int32_t, half>(dd, numpyArray,
    //         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  }*/ else if (
        dtype.is(py::dtype::of<float>())) {
      fillDescriptorDataImpl<float, float>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    } else if (dtype.is(py::dtype::of<double>())) {
      fillDescriptorDataImpl<double, double>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
    } else if (dtype.is(py::dtype::of<std::complex<float>>())) {
      fillDescriptorDataImpl<std::complex<float>, std::complex<float>>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64);
    } else if (dtype.is(py::dtype::of<std::complex<double>>())) {
      fillDescriptorDataImpl<std::complex<double>, std::complex<double>>(
          dd, numpyArray, ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128);
    } else {
      throw std::runtime_error("Unsupported numpy data type");
    }

    ++ddIterator;
    ++npIterator;
  }
}

template <typename onnx_type, typename py_type>
void DataConversion::addNumpyArray(py::list& numpyArrayList,
                                   const DescriptorData& dd) {
  auto numElements = std::accumulate(dd.shape.begin(), dd.shape.end(), 1UL,
                                     std::multiplies<uint64_t>());
  py_type intermediateBuffer[numElements];
  onnx_type* oldBuffer = reinterpret_cast<onnx_type*>(dd.descriptor.buffer);
  std::copy(oldBuffer, oldBuffer + numElements, intermediateBuffer);
  std::vector<ptrdiff_t> shape(dd.descriptor.shape,
                               dd.descriptor.shape + dd.descriptor.dimensions);
  numpyArrayList.append(
      std::move(py::array_t<py_type>(shape, intermediateBuffer)));
}

void DataConversion::fillNumpyArrayList(
    py::list& numpyArrayList,
    const std::vector<DescriptorData>& descriptorsData) {
  for (const auto& dd : descriptorsData) {
    DISPATCH_OVER_NUMERIC_DATA_TYPE(dd.descriptor.dataType, addNumpyArray,
                                    numpyArrayList, dd)
  }
}

std::vector<onnxTensorDescriptorV1> DataConversion::getTensorDescriptors(
    const std::vector<DescriptorData>& descriptorsData) {
  std::vector<onnxTensorDescriptorV1> tensorDescriptors;
  for (const auto& dd : descriptorsData) {
    tensorDescriptors.emplace_back(dd.descriptor);
  }
  return tensorDescriptors;
}

py::list DataConversion::getOutputs() const {
  py::list outputs;
  fillNumpyArrayList(outputs, output_descriptors_data_);
  return outputs;
}

void DataConversion::setInputs(const py::list& inputs) {
  fillDescriptorDataVector(inputs, input_descriptors_data_);
}

std::vector<onnxTensorDescriptorV1> DataConversion::getInputDescriptors()
    const {
  return getTensorDescriptors(input_descriptors_data_);
}

std::vector<onnxTensorDescriptorV1> DataConversion::getOutputDescriptors()
    const {
  return getTensorDescriptors(output_descriptors_data_);
}
