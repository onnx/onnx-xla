#include "onnx/onnxifi.h"
#include "onnx_xla/onnxifi_helper.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

// TODO: Figure out how to determine type of device, what information to store
//      about hardware, and how to modify execution as a result

onnxStatus onnxifiTryCatch(std::function<onnxStatus()> tryBlock) {
  try {
    return tryBlock();
  } catch (const std::bad_alloc& e) {
    std::cerr << "Allocation failed: " << e.what() << std::endl;
    return ONNXIFI_STATUS_NO_SYSTEM_MEMORY;
  } catch (const std::exception& e) {
    std::cerr << "Internal Error: " << e.what() << std::endl;
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  } catch (...) {
    std::cerr << "Internal Error" << std::endl;
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }
}

// Create 1 backendID
// TODO: Determining # of CPU, GPU, TPU devices and return
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendIDs(onnxBackendID* backendIDs, size_t* numBackends) {
  return onnxifiTryCatch([&] {
    if (!numBackends) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (!backendIDs || *numBackends < 1) {
      *numBackends = 1;
      return ONNXIFI_STATUS_FALLBACK;
    }
    *backendIDs = reinterpret_cast<onnxBackendID>(new OnnxXlaBackendID());
    *numBackends = 1;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

// Free memory for given backend ID
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseBackendID(onnxBackendID backendID) {
  return onnxifiTryCatch([&] {
    if (!backendID) {
      return ONNXIFI_STATUS_INVALID_ID;
    }
    auto* backend_id = reinterpret_cast<OnnxXlaBackendID*>(backendID);
    delete backend_id;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

// Returning info for given ID
// TODO: Make sure this information is correct
// TODO: Expand for different IDs (TPU/GPU) in the future
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendInfo(onnxBackendID backendID,
                   onnxBackendInfo infoType,
                   void* infoValue,
                   size_t* infoValueSize) {
  return onnxifiTryCatch([&] {
    if (!infoValueSize) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }

    if (!backendID) {
      return ONNXIFI_STATUS_INVALID_ID;
    }

    auto SET_STRING = [&](const char* str) {
      if (!infoValue || *infoValueSize < strlen(str) + 1) {
        *infoValueSize = strlen(str) + 1;
        return ONNXIFI_STATUS_FALLBACK;
      }
      strncpy((char*)(infoValue), str, *infoValueSize);
      *infoValueSize = strlen(str) + 1;
      return ONNXIFI_STATUS_SUCCESS;
    };
    auto SET_UINT64 = [&](uint64_t x) {
      if (!infoValue || *infoValueSize < sizeof(uint64_t)) {
        *infoValueSize = sizeof(uint64_t);
        return ONNXIFI_STATUS_FALLBACK;
      }
      *(uint64_t*)(infoValue) = x;
      *infoValueSize = sizeof(uint64_t);
      return ONNXIFI_STATUS_SUCCESS;
    };

    switch (infoType) {
      case ONNXIFI_BACKEND_NAME: {
        return SET_STRING("onnx-xla");
      }
      case ONNXIFI_BACKEND_VENDOR: {
        return SET_STRING("Google");
      }
      case ONNXIFI_BACKEND_VERSION: {
        return SET_STRING("1.0.0");
      }
      case ONNXIFI_BACKEND_EXTENSIONS: {
        *infoValueSize = 0;
        return ONNXIFI_STATUS_SUCCESS;
      }
      case ONNXIFI_BACKEND_DEVICE: {
        return SET_STRING("cpu (for now in development)");
      }
      case ONNXIFI_BACKEND_DEVICE_TYPE: {
        return SET_UINT64(ONNXIFI_DEVICE_TYPE_CPU);
      }
      case ONNXIFI_BACKEND_CAPABILITIES: {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_INIT_PROPERTIES: {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_MEMORY_TYPES: {
        return SET_UINT64(ONNXIFI_MEMORY_TYPE_CPU);
      }
      case ONNXIFI_BACKEND_MEMORY_SIZE: {
        // TODO
        return ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE;
      }
      case ONNXIFI_BACKEND_MAX_GRAPH_SIZE: {
        return SET_UINT64(1000000UL);
      }
      case ONNXIFI_BACKEND_MAX_GRAPH_COUNT: {
        return SET_UINT64(1UL);
      }
      case ONNXIFI_BACKEND_MACS_FP32: {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_MACS_FP16: {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_MEMORY_BANDWIDTH: {
        return SET_UINT64(0UL);
      }
      case ONNXIFI_BACKEND_CPU_MEMORY_READ_BANDWIDTH: {
        return SET_UINT64(0UL);
      }
      default: { return ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE; }
    }
  });
}

// TODO: Figure out how to get compatibility e.g. sufficient to run OnnxParser?
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendCompatibility(onnxBackendID backendID,
                            size_t onnxModelSize,
                            const void* onnxModel) {
  return onnxifiTryCatch([&] {
    if (!onnxModel) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (onnxModelSize == 0) {
      return ONNXIFI_STATUS_INVALID_SIZE;
    }
    return ONNXIFI_STATUS_SUCCESS;
  });
}

// TODO: any arguments to pass?
// Create and return a BackendControl object
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxInitBackend(onnxBackendID backendID,
                const uint64_t* auxPropertiesList,
                onnxBackend* backend) {
  return onnxifiTryCatch([&] {
    auto* backend_id = reinterpret_cast<OnnxXlaBackendID*>(backendID);
    *backend = reinterpret_cast<onnxBackend>(new BackendControl(backend_id));
    return ONNXIFI_STATUS_SUCCESS;
  });
}

// Release BackendControl object
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseBackend(onnxBackend backend) {
  return onnxifiTryCatch([&] {
    if (!backend) {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }
    auto* backendController = reinterpret_cast<BackendControl*>(backend);
    delete backendController;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

// Create and return XlaExecutor object
// TODO: Ignore the weightDescriptors for now and rely on initialization list
// TODO: more robust error handling in header file to be included
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxInitGraph(onnxBackend backend,
              const uint64_t* auxPropertiesList,
              size_t onnxModelSize,
              const void* onnxModel,
              uint32_t weightsCount,
              const onnxTensorDescriptorV1* weightDescriptors,
              onnxGraph* graph) {
  return onnxifiTryCatch([&] {
    *graph = NULL;
    if (!backend) {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }
    if (!onnxModel) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (onnxModelSize == 0) {
      return ONNXIFI_STATUS_INVALID_SIZE;
    }
    auto* backendController = reinterpret_cast<BackendControl*>(backend);
    return backendController->build(onnxModel, onnxModelSize, weightsCount,
                                    weightDescriptors, graph);
  });
}

// Verify IO metadata and use initIO to store location of IO
// TODO: memoryType field ignored for now
// TODO: more robust error handling in header file to be included
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxSetGraphIO(onnxGraph graph,
               uint32_t inputsCount,
               const onnxTensorDescriptorV1* inputDescriptors,
               uint32_t outputsCount,
               const onnxTensorDescriptorV1* outputDescriptors) {
  return onnxifiTryCatch([&] {
    if (!graph) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    if (!inputDescriptors || !outputDescriptors) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    auto* executor = reinterpret_cast<onnx_xla::XlaExecutor*>(graph);
    return executor->initIO(inputsCount, inputDescriptors, outputsCount,
                            outputDescriptors);
  });
}

// Runs the XlaExecutor by sending literals to server and executing computation
// TODO: support for synchronization primitives; For now assume, they are always
// set
// TODO: more robust error handling in header file to be included
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxRunGraph(onnxGraph graph,
             const onnxMemoryFenceV1* inputFence,
             onnxMemoryFenceV1* outputFence) {
  return onnxifiTryCatch([&] {
    if (!graph) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    // TODO: Status code specific to events
    // TODO: Inform user that only ONNXIFI_SYNCHRONIZATION_EVENT is the only
    // acceptable type
    if (!inputFence) {
      throw std::runtime_error("Invalid input memory fence");
    }
    if (inputFence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1) {
      return ONNXIFI_STATUS_UNSUPPORTED_TAG;
    }
    if (inputFence->type != ONNXIFI_SYNCHRONIZATION_EVENT) {
      throw std::runtime_error(
          "The input memory fence must have type "
          "ONNXIFI_SYNCHRONIZATION_EVENT. "
          "The event must be initialized.");
    }
    if (!outputFence) {
      throw std::runtime_error("Invalid output memory fence");
    }
    if (outputFence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1) {
      return ONNXIFI_STATUS_UNSUPPORTED_TAG;
    }
    if (outputFence->type != ONNXIFI_SYNCHRONIZATION_EVENT) {
      throw std::runtime_error(
          "The output memory fence must have type "
          "ONNXIFI_SYNCHRONIZATION_EVENT. "
          "The event cannot be initialized.");
    }

    auto* executor = reinterpret_cast<onnx_xla::XlaExecutor*>(graph);
    auto initStatus = onnxInitEvent(executor->backend_, &outputFence->event);
    if (initStatus != ONNXIFI_STATUS_SUCCESS) {
      return initStatus;
    }
    return executor->executeComputation(inputFence, outputFence);
  });
}

// Frees executor memory
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseGraph(onnxGraph graph) {
  return onnxifiTryCatch([&] {
    if (!graph) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    auto* executor = reinterpret_cast<onnx_xla::XlaExecutor*>(graph);
    delete executor;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

// Returns event state
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetEventState(onnxEvent event, onnxEventState* state) {
  if (!state) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  *state = ONNXIFI_EVENT_STATE_INVALID;
  if (!event) {
    return ONNXIFI_STATUS_INVALID_EVENT;
  }
  auto* eventController = reinterpret_cast<EventControl*>(event);
  {
    std::lock_guard<std::mutex> lk(eventController->mutex_);
    if (eventController->signalled_) {
      *state = ONNXIFI_EVENT_STATE_SIGNALLED;
    } else {
      *state = ONNXIFI_EVENT_STATE_NONSIGNALLED;
    }
  }
  return ONNXIFI_STATUS_SUCCESS;
}

// Initialize event by creating EventControl object on the heap
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxInitEvent(onnxBackend backend, onnxEvent* event) {
  return onnxifiTryCatch([&] {
    if (!event) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    *event = NULL;

    if (!backend) {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }

    *event = reinterpret_cast<onnxEvent>(new EventControl());
    return ONNXIFI_STATUS_SUCCESS;
  });
}

// Signal Event by changing the signalled boolean under mutex hold
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxSignalEvent(onnxEvent event) {
  return onnxifiTryCatch([&] {
    if (!event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    auto* eventController = reinterpret_cast<EventControl*>(event);
    {
      std::lock_guard<std::mutex> lk(eventController->mutex_);
      if (eventController->signalled_) {
        return ONNXIFI_STATUS_INVALID_STATE;
      }
      eventController->signalled_ = true;
    }
    eventController->condvar_.notify_all();
    return ONNXIFI_STATUS_SUCCESS;
  });
}

// Wait for signalled to be turned true using conditional variable to coordinate
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxWaitEvent(onnxEvent event) {
  return onnxifiTryCatch([&] {
    if (!event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }

    auto* eventController = reinterpret_cast<EventControl*>(event);
    std::unique_lock<std::mutex> lk(eventController->mutex_);
    eventController->condvar_.wait(
        lk, [&eventController] { return eventController->signalled_; });

    return ONNXIFI_STATUS_SUCCESS;
  });
}

// Free memory that was allocated for the EventControl object
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseEvent(onnxEvent event) {
  return onnxifiTryCatch([&] {
    if (!event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    auto* eventController = reinterpret_cast<EventControl*>(event);
    delete eventController;
    return ONNXIFI_STATUS_SUCCESS;
  });
}
