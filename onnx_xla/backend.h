#pragma once

#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"
#include "onnx/onnxifi.h"
#include "onnx/proto_utils.h"
#include "onnx/shape_inference/implementation.h"

#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"
#include <grpcpp/grpcpp.h>

#include "onnx_xla/utils.h"
#include "onnx_xla/operator_registry.h"

#include <memory>

namespace onnx_xla {
using ::xla::GlobalData;
using ::xla::XlaComputation;
using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::Tensor;
using ::ONNX_NAMESPACE::Graph;

class XlaTransform;
class XlaExecutor;
class OnnxParser;

// Engine to execute an XlaComputation constructed by XlaTransform. The
// computation_ is filled by the XlaTransform object. To run, call initIO
// to verify IO metadata and to declare IO locations. Once IO data is
// present, execute executeComputation to run. If successful, output
// tensors will be present at the output_buffers_ pointers.

class XlaExecutor final {
 public:
  // Constructor initialized with backend handle
  XlaExecutor(onnxBackend backend);

  // Used to pass IO metadata and locations to the engine
  onnxStatus initIO(uint32_t inputsCount,
                    const onnxTensorDescriptorV1* inputDescriptors,
                    uint32_t outputsCount,
                    const onnxTensorDescriptorV1* outputDescriptors);

  // Sends input tensor values to the server
  // Input fence (initialized) signals when inputs are ready
  // Runs the computation on the server using passed input
  // outputFence (initialized) is signalled once outputs are ready
  onnxStatus executeComputation(const onnxMemoryFenceV1* inputFence,
                                onnxMemoryFenceV1* outputFence);

  // backend handle
  const onnxBackend backend_;

 private:
  // computation to be run
  XlaComputation computation_;

  // Store IO metadata to
  //  Verify IO has correct shape, data type, TODO: memory type
  //  Get input and output locations
  uint32_t num_inputs_;
  uint32_t num_outputs_;
  std::unordered_map<std::string, ONNX_NAMESPACE::TensorProto_DataType>
      io_data_type_;
  std::unordered_map<std::string, std::vector<Dimension>> io_shape_;
  std::unordered_map<std::string, onnxPointer> input_buffers_;
  std::unordered_map<std::string, onnxPointer> output_buffers_;

  // Mapping of parameter number to input name; use to fill arguments_ in the
  // correct order
  std::vector<std::string> param_input_name_;

  // Used to copy output returned from XLA to output buffers
  std::vector<std::string> output_names_;

  // Helper functions to translate tensors, inputs, and weights to literals
  std::unique_ptr<Literal> tensorToLiteral(const Tensor& t);
  std::unique_ptr<Literal> inputNameToLiteral(const std::string& name);
  std::unique_ptr<Literal> descriptorToLiteral(const onnxTensorDescriptorV1& t);

  friend class XlaTransform;
};

// Engine to transform an IR graph to a form that can be executed by the XLA
// server.
// When the object is constructed, ownership of the IR graph is passed to it.
// Then,
// execute translateGraph to build up the XlaExecutor object (and thus the
// XlaComputation).
// To get the handle on the XlaExecutor that is constructed, call executor()
// (can only be
// done once).
class XlaTransform final {
 public:
  // Passes IR graph to be transformed, name of builder, and weightDescriptor
  // info
  // TODO: Remove build_name? or keep for debugging purposes?
  XlaTransform(onnxBackend backend,
               std::unique_ptr<Graph> ir,
               const std::string& build_name,
               uint32_t weightsCount,
               const onnxTensorDescriptorV1* weightDescriptors);
  ~XlaTransform();

  // Fills up XlaExecutor based on the IR graph. Function accomplishes:
  //  Initializer/weight values added as constants to the graph
  //  Fills up executor_'s expected IO metadata, which can be verified in initIO
  //  Translates IR graph node by node dispatching to operator registry
  //    TODO: Fix kUndefinded translation, which is present for relu test
  //  Fills up exector_'s output names
  //  Returns status
  onnxStatus translateGraph();

  // Used to get handle to XlaExecutor
  // NOTE: Can only be called on once as it releases the unique pointer. Freeing
  // memory must be handled by caller.
  XlaExecutor* executor();

 private:
  // IR graph to be translated
  std::unique_ptr<Graph> ir_;

  // Weight Descriptor information
  uint32_t weights_count_;
  const onnxTensorDescriptorV1* weight_descriptors_;

  // Builder that builds XlaComputation
  //  TODO: Remove? Currently only used by one function
  XlaBuilder builder_;

  // XlaExecutor that is built up to do the computation later
  std::unique_ptr<XlaExecutor> executor_;

  // Used to keep track of values and XlaOp's
  // whose output corresponds to them
  ValueOpMap value_to_op_;

  // Keep track of constant literals
  ValueLiteralMap value_to_literal_;

  // Keeps track of number of parameters in computation
  //  TODO: Make local? Only used by one function
  int64 global_param_number_;

  // Helper to get shape of associated value
  static inline Shape shapeOfValue(const Value* v);

  // Create ConstantLiteral XlaOps for initializers/weights, verifying weight
  // descriptors;
  // Creates params for other runtime inputs
  // Fill executor_'s input metadata (type, shape) to be verified later
  onnxStatus handleInputs();

  // Fill output_names_
  // Create output XlaOp
  // Fill executor_'s output metadata (type, shape) to be verified later
  onnxStatus handleOutputs();
};

// Engine to build up an IR graph from proto bytes format. Model validation and
// shape inference for entire graph is conducted here.
class OnnxParser {
 public:
  // Initializes parser
  OnnxParser(const void* serializedModel, size_t serializedModelSize);

  // Deserialize to modelProto, shape inference, and conversion to IR
  //(model validation) stored in ir
  onnxStatus parse(std::unique_ptr<Graph>& ir);

 private:
  const void* serialized_model_;
  size_t serialized_model_size_;
};
}
