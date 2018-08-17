#include "onnx_xla/backend.h"

namespace onnx_xla {
XlaExecutor::XlaExecutor(onnxBackend backend) : backend_(backend) {}

#define SWITCH(data_type)                                                 \
  switch (data_type) {                                                    \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {                    \
      OPERATION(float, float, floats)                                     \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {                \
      OPERATION(complex64, complex64, floats)                             \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {                  \
      OPERATION(int32_t, half, int32s)                                    \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL: {                     \
      OPERATION(int32_t, bool, int32s)                                    \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {                     \
      OPERATION(int32_t, int8, int32s)                                    \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {                    \
      OPERATION(int32_t, int16, int32s)                                   \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {                    \
      OPERATION(int32_t, int32, int32s)                                   \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {                    \
      OPERATION(int32_t, uint8, int32s)                                   \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {                   \
      OPERATION(int32_t, uint16, int32s)                                  \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {                    \
      OPERATION(int64_t, int64, int64s)                                   \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {                   \
      OPERATION(uint64_t, uint32, uint64s)                                \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {                   \
      OPERATION(uint64_t, uint64, uint64s)                                \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {                   \
      OPERATION(double, double, doubles)                                  \
    }                                                                     \
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:                 \
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:                     \
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:                  \
    default: {                                                            \
      throw std::runtime_error("Tensor not of a convertible data type."); \
    }                                                                     \
  }

std::unique_ptr<Literal> XlaExecutor::tensorToLiteral(const Tensor& t) {
#define OPERATION(type_from, type_to, vec)                                   \
  type_from* t_data;                                                         \
  if (t.is_raw_data()) {                                                     \
    t_data = (type_from*)t.raw().c_str();                                    \
  } else {                                                                   \
    t_data = (type_from*)t.vec().data();                                     \
  }                                                                          \
  std::vector<int64> sizes;                                                  \
  for (auto n : t.sizes()) {                                                 \
    sizes.push_back(n);                                                      \
  }                                                                          \
  auto l = std::unique_ptr<Literal>(new Literal(                             \
      ShapeUtil::MakeShape(NativeToPrimitiveType<type_to>(), sizes)));       \
  int64 num_elements = std::accumulate(sizes.begin(), sizes.end(), (int64)1, \
                                       std::multiplies<int64>());            \
  tensorflow::gtl::MutableArraySlice<type_to> l_data = l->data<type_to>();   \
  for (auto i = 0; i < num_elements; ++i) {                                  \
    l_data[i] = (type_to)t_data[i];                                          \
  }                                                                          \
  return l;

  SWITCH(t.elem_type())
#undef OPERATION
}

std::unique_ptr<Literal> XlaExecutor::inputNameToLiteral(
    const std::string& name) {
  std::vector<int64> sizes;
  for (auto i = 0; i < io_shape_[name].size(); ++i) {
    sizes.push_back((int64)io_shape_[name][i].dim);
  }
  int64 num_elements = std::accumulate(sizes.begin(), sizes.end(), (int64)1,
                                       std::multiplies<int64>());

#define OPERATION(type_from, type_to, vec)                                 \
  auto l = std::unique_ptr<Literal>(new Literal(                           \
      ShapeUtil::MakeShape(NativeToPrimitiveType<type_to>(), sizes)));     \
  tensorflow::gtl::MutableArraySlice<type_to> l_data = l->data<type_to>(); \
  ONNX_ASSERT(input_buffers_[name]);                                       \
  type_from* inputData = (type_from*)input_buffers_[name];                 \
  for (auto i = 0; i < num_elements; ++i) {                                \
    l_data[i] = (type_to)inputData[i];                                     \
  }                                                                        \
  return l;

  SWITCH(io_data_type_[name])
#undef OPERATION
}

std::unique_ptr<Literal> XlaExecutor::descriptorToLiteral(
    const onnxTensorDescriptorV1& t) {
  std::vector<int64> sizes(&t.shape[0], &t.shape[t.dimensions]);
  int64 num_elements = std::accumulate(sizes.begin(), sizes.end(), (int64)1,
                                       std::multiplies<int64>());

#define OPERATION(type_from, type_to, vec)                                 \
  auto l = std::unique_ptr<Literal>(new Literal(                           \
      ShapeUtil::MakeShape(NativeToPrimitiveType<type_to>(), sizes)));     \
  tensorflow::gtl::MutableArraySlice<type_to> l_data = l->data<type_to>(); \
  type_from* inputData = (type_from*)t.buffer;                             \
  for (auto i = 0; i < num_elements; ++i) {                                \
    l_data[i] = (type_to)inputData[i];                                     \
  }                                                                        \
  return l;

  SWITCH(t.dataType)
#undef OPERATION
}

onnxStatus XlaExecutor::initIO(
    uint32_t inputsCount,
    const onnxTensorDescriptorV1* inputDescriptors,
    uint32_t outputsCount,
    const onnxTensorDescriptorV1* outputDescriptors) {
  if (num_inputs_ != inputsCount) {
    throw std::runtime_error("Did not receive expected number of inputs");
  }
  if (num_outputs_ != outputsCount) {
    throw std::runtime_error("Did not receive expected number of outputs");
  }

#define CHECK_TYPE_AND_SHAPE(VAR)                                      \
  for (auto i = 0; i < num_##VAR##s_; ++i) {                           \
    if (VAR##Descriptors[i].tag != ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1) { \
      return ONNXIFI_STATUS_UNSUPPORTED_TAG;                           \
    }                                                                  \
    const std::string name(VAR##Descriptors[i].name);                  \
    if (io_data_type_.find(name) == io_data_type_.end()) {             \
      return ONNXIFI_STATUS_INVALID_NAME;                              \
    }                                                                  \
    VAR##_buffers_[name] = VAR##Descriptors[i].buffer;                 \
    if (VAR##Descriptors[i].dataType != io_data_type_[name]) {         \
      return ONNXIFI_STATUS_MISMATCHING_DATATYPE;                      \
    }                                                                  \
    if (VAR##Descriptors[i].dimensions != io_shape_[name].size()) {    \
      return ONNXIFI_STATUS_MISMATCHING_SHAPE;                         \
    }                                                                  \
    for (auto j = 0; j < io_shape_[name].size(); ++j) {                \
      if (!io_shape_[name][j].is_int ||                                \
          io_shape_[name][j].dim != VAR##Descriptors[i].shape[j]) {    \
        return ONNXIFI_STATUS_MISMATCHING_SHAPE;                       \
      }                                                                \
    }                                                                  \
  }

  CHECK_TYPE_AND_SHAPE(input);
  CHECK_TYPE_AND_SHAPE(output);
  return ONNXIFI_STATUS_SUCCESS;
#undef CHECK_TYPE_AND_SHAPE
}

onnxStatus XlaExecutor::executeComputation(const onnxMemoryFenceV1* inputFence,
                                           onnxMemoryFenceV1* outputFence) {
  std::vector<GlobalData*> arguments;
  auto waitStatus = onnxWaitEvent(inputFence->event);
  if (waitStatus != ONNXIFI_STATUS_SUCCESS) {
    return waitStatus;
  }
  for (const std::string& s : param_input_name_) {
    auto l_ptr = this->inputNameToLiteral(s);
    auto l_data_ptr = xla::TransferParameterToServer(*l_ptr);
    arguments.push_back(l_data_ptr.release());
  }
  auto result = xla::ExecuteComputation(computation_, arguments);

#define OPERATION(type_to, type_from, vec)                            \
  type_to* destination = (type_to*)output_buffers_[output_names_[i]]; \
  for (auto j = 0; j < num_elements; ++j) {                           \
    destination[j] = (type_to)outputLiterals[i].data<type_from>()[j]; \
  }                                                                   \
  break;

  std::vector<Literal> outputLiterals = result->DecomposeTuple();
  for (auto i = 0; i < outputLiterals.size(); ++i) {
    int64_t num_elements = 1;
    for (auto j = 0; j < io_shape_[output_names_[i]].size(); ++j) {
      num_elements *= io_shape_[output_names_[i]][j].dim;
    }
    SWITCH(io_data_type_[output_names_[i]])
  }
  return onnxSignalEvent(outputFence->event);
#undef OPERATION
}

XlaTransform::XlaTransform(onnxBackend backend,
                           std::unique_ptr<Graph> ir,
                           const std::string& build_name,
                           uint32_t weightsCount,
                           const onnxTensorDescriptorV1* weightDescriptors)
    : weights_count_(weightsCount),
      weight_descriptors_(weightDescriptors),
      builder_(build_name),
      executor_(new XlaExecutor(backend)),
      global_param_number_(0) {
  ir_ = std::move(ir);
}

XlaTransform::~XlaTransform() {}

inline Shape XlaTransform::shapeOfValue(const Value* v) {
  std::vector<int64> sizes;
  for (const Dimension& d : v->sizes()) {
    ONNX_ASSERT(d.is_int);
    sizes.push_back(d.dim);
  }
  return ShapeUtil::MakeShape(onnxToPrimitive(v->elemType()), sizes);
}

onnxStatus XlaTransform::handleInputs() {
  if (ir_->initializers().size() != 0 && weight_descriptors_) {
    throw std::runtime_error(
        "Static weights of the graph should be passed through "
        "ModelProto.graph.initializer,"
        "or through the weightDescriptors parameters, not both");
  }
  if (weights_count_ > 0 && !weight_descriptors_) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  std::unordered_map<std::string, const Value*> inputNameToValue;
  for (const Value* v : ir_->inputs()) {
    inputNameToValue[v->uniqueName()] = v;
  }
  std::unordered_map<std::string, bool> isInitialized;

  if (weight_descriptors_) {
    executor_->num_inputs_ =
        (uint32_t)((int64_t)(ir_->inputs().size()) - (int64_t)(weights_count_));
    for (auto i = 0; i < weights_count_; ++i) {
      if (weight_descriptors_[i].tag != ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1) {
        return ONNXIFI_STATUS_UNSUPPORTED_TAG;
      }
      std::string name(weight_descriptors_[i].name);
      isInitialized[name] = true;
      const onnxTensorDescriptorV1& t = weight_descriptors_[i];
      const Value* v = inputNameToValue[name];
      if (t.dataType == v->elemType()) {
        return ONNXIFI_STATUS_MISMATCHING_DATATYPE;
      }
      if (t.memoryType != ONNXIFI_MEMORY_TYPE_CPU) {
        throw std::runtime_error(
            "The weightDescriptors parameters must have"
            "memoryType ONNXIFI_MEMORY_TYPE_CPU");
      }
      if (t.dimensions != v->sizes().size()) {
        return ONNXIFI_STATUS_MISMATCHING_SHAPE;
      }
      for (auto j = 0; j < t.dimensions; ++j) {
        if (!v->sizes()[j].is_int || t.shape[j] != v->sizes()[j].dim) {
          return ONNXIFI_STATUS_MISMATCHING_SHAPE;
        }
      }
      auto l_ptr = executor_->descriptorToLiteral(t);
      auto constant = builder_.ConstantLiteral(*l_ptr);
      value_to_op_[v] = constant;
      value_to_literal_[v] = std::move(l_ptr);
    }
  } else {
    executor_->num_inputs_ = (uint32_t)((int64_t)(ir_->inputs().size()) -
                                        (int64_t)(ir_->initializers().size()));
    for (const Tensor& t : ir_->initializers()) {
      std::string name(t.name());
      isInitialized[name] = true;
      auto l_ptr = executor_->tensorToLiteral(t);
      auto constant = builder_.ConstantLiteral(*l_ptr);
      const Value* v = inputNameToValue[name];
      value_to_op_[v] = constant;
      value_to_literal_[v] = std::move(l_ptr);
    }
  }
  for (const Value* v : ir_->inputs()) {
    if (isInitialized.find(v->uniqueName()) == isInitialized.end()) {
      executor_->param_input_name_.push_back(v->uniqueName());
      auto param = builder_.Parameter(global_param_number_++, shapeOfValue(v),
                                      v->uniqueName());
      value_to_op_[v] = param;
      executor_->io_data_type_[v->uniqueName()] = v->elemType();
      executor_->io_shape_[v->uniqueName()] = v->sizes();
    }
  }
  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus XlaTransform::handleOutputs() {
  executor_->num_outputs_ = (uint32_t)ir_->outputs().size();
  std::vector<XlaOp> retOps;
  for (const Value* v : ir_->outputs()) {
    executor_->io_data_type_[v->uniqueName()] = v->elemType();
    executor_->io_shape_[v->uniqueName()] = v->sizes();
    retOps.push_back(value_to_op_[v]);
    executor_->output_names_.push_back(v->uniqueName());
  }
  builder_.Tuple(retOps);
  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus XlaTransform::translateGraph() {
  auto handleInputsStatus = this->handleInputs();
  if (handleInputsStatus != ONNXIFI_STATUS_SUCCESS) {
    return handleInputsStatus;
  }
  auto& registry = OperatorRegistry::registry();
  for (auto it = ir_->begin(); it != ir_->end(); ++it) {
    auto translateStatus =
        registry.translate(**it, builder_, value_to_op_, value_to_literal_);
    if (translateStatus != ONNXIFI_STATUS_SUCCESS) {
      return translateStatus;
    }
  }

  auto handleOutputsStatus = this->handleOutputs();
  if (handleOutputsStatus != ONNXIFI_STATUS_SUCCESS) {
    return handleOutputsStatus;
  }
  auto computation_status = builder_.Build();
  if (!computation_status.ok()) {
    throw std::runtime_error("The graph was not able to be built");
  }
  executor_->computation_ = computation_status.ConsumeValueOrDie();
  return ONNXIFI_STATUS_SUCCESS;
}

XlaExecutor* XlaTransform::executor() {
  return executor_.release();
}

OnnxParser::OnnxParser(const void* serializedModel, size_t serializedModelSize)
    : serialized_model_(serializedModel),
      serialized_model_size_(serializedModelSize) {}

onnxStatus OnnxParser::parse(std::unique_ptr<Graph>& ir) {
  ModelProto deserializedModel;
  if (!ONNX_NAMESPACE::ParseProtoFromBytes(&deserializedModel,
                                           (const char*)serialized_model_,
                                           serialized_model_size_)) {
    return ONNXIFI_STATUS_INVALID_PROTOBUF;
  }
  try {
    ONNX_NAMESPACE::shape_inference::InferShapes(deserializedModel);
    ir = ONNX_NAMESPACE::ImportModelProto(deserializedModel);
    return ONNXIFI_STATUS_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return ONNXIFI_STATUS_INVALID_MODEL;
  } catch (...) {
    return ONNXIFI_STATUS_INVALID_MODEL;
  }
}
#undef SWITCH
}
