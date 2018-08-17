#pragma once

#include "backend.h"

// TODO: More formal representation of backendID - CPU, GPU, TPU?
struct OnnxXlaBackendID {
  int device_id{0};
};

struct EventControl {
  EventControl();
  volatile bool signalled_;
  std::mutex mutex_;
  std::condition_variable condvar_;
};

// Backend engine
//  backendID will eventually determine translation detail
struct BackendControl {
 public:
  BackendControl(OnnxXlaBackendID* id);

  // use OnnxParser and XlaTransform to fill *graph
  onnxStatus build(const void* serializedModel,
                   size_t serializedModelSize,
                   uint32_t weightsCount,
                   const onnxTensorDescriptorV1* weightDescriptors,
                   onnxGraph* graph);

 private:
  OnnxXlaBackendID* backendID;
};
