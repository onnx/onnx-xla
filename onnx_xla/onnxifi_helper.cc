#include "onnxifi_helper.h"

EventControl::EventControl() : signalled_(false) {}

BackendControl::BackendControl(OnnxXlaBackendID* id) : backendID(id) {}

onnxStatus BackendControl::build(
    const void* serializedModel,
    size_t serializedModelSize,
    uint32_t weightsCount,
    const onnxTensorDescriptorV1* weightDescriptors,
    onnxGraph* graph) {
  onnx_xla::OnnxParser parser(serializedModel, serializedModelSize);
  std::unique_ptr<ONNX_NAMESPACE::Graph> ir(nullptr);
  auto parseStatus = parser.parse(ir);
  if (parseStatus != ONNXIFI_STATUS_SUCCESS) {
    return parseStatus;
  }
  std::string build_name = ir->name();
  onnx_xla::XlaTransform runner(reinterpret_cast<onnxBackend>(this),
                                std::move(ir), build_name, weightsCount,
                                weightDescriptors);
  auto translateStatus = runner.translateGraph();
  if (translateStatus != ONNXIFI_STATUS_SUCCESS) {
    return translateStatus;
  }

  *graph = reinterpret_cast<onnxGraph>(runner.executor());
  return ONNXIFI_STATUS_SUCCESS;
}
