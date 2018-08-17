#include "onnx_xla/onnxifi_helper.h"
#include "onnx_xla/backend_test.h"
#include <stdlib.h>
#include <cmath>

namespace onnx_xla {

bool almost_equal(float a, float b, float epsilon) {
  return std::abs(a - b) < epsilon;
}

void static_relu_test() {
  // Set up IR graph
  std::unique_ptr<Graph> relu_graph(new Graph());
  relu_graph->setName("relu_graph");
  Tensor initializer;
  initializer.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  std::uniform_real_distribution<float> unif(-0.5, 0.5);
  for (int i = 0; i < 24; ++i) {
    std::random_device rand_dev;
    std::mt19937 rand_engine(rand_dev());
    initializer.floats().push_back(unif(rand_engine));
  }
  initializer.sizes().push_back(2);
  initializer.sizes().push_back(3);
  initializer.sizes().push_back(4);
  std::vector<Dimension> sizes;
  sizes.push_back(2);
  sizes.push_back(3);
  sizes.push_back(4);
  relu_graph->addInitializerAndInput(initializer, "relu_input");
  auto relu_node = relu_graph->create(Symbol("Relu"), relu_graph->inputs());
  relu_graph->appendNode(relu_node);
  auto relu_output = relu_node->output();
  relu_output->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  relu_output->setSizes(sizes);
  relu_output->setUniqueName("relu_output");
  relu_graph->return_node()->addInput(relu_output);

  // Set up IO information
  uint64_t shape[3] = {2, 3, 4};
  onnxTensorDescriptorV1 output;
  output.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  output.name = "relu_output";
  output.dataType = ONNXIFI_DATATYPE_FLOAT32;
  output.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  output.dimensions = 3;
  output.shape = shape;
  output.buffer = (onnxPointer) new float[24];

  // Setup events
  // Hacky event usage to make it work (cannot use onnxifi with backend)
  onnxMemoryFenceV1 inputFence;
  inputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
  inputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
  auto inputEvent = new EventControl();
  inputEvent->signalled_ = true;
  inputFence.event = reinterpret_cast<onnxEvent>(inputEvent);
  onnxMemoryFenceV1 outputFence;
  outputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
  outputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
  auto outputEvent = new EventControl();
  outputEvent->signalled_ = false;
  outputFence.event = reinterpret_cast<onnxEvent>(outputEvent);

  // Execute using XLA backend
  XlaTransform runner(NULL, std::move(relu_graph), "relu", 0, nullptr);
  runner.translateGraph();
  auto executor = runner.executor();
  executor->initIO(0, nullptr, 1, &output);
  executor->executeComputation(&inputFence, &outputFence);

  // Check correctness
  ONNX_ASSERT(outputEvent->signalled_);
  float* output_ptr = (float*)output.buffer;
  for (int i = 0; i < 24; ++i) {
    if (initializer.floats()[i] > 0.0f) {
      ONNX_ASSERT(almost_equal(initializer.floats()[i], output_ptr[i]));
    } else {
      ONNX_ASSERT(almost_equal(0.0f, output_ptr[i]));
    }
  }

  // Free memory
  delete executor;
  delete[] output_ptr;
  delete inputEvent;
  delete outputEvent;
}

void dynamic_relu_test() {
  // Set up IR graph
  std::unique_ptr<Graph> relu_graph(new Graph());
  relu_graph->setName("relu_graph");
  Value* relu_input = relu_graph->addInput();
  std::vector<Dimension> sizes;
  sizes.push_back(2);
  sizes.push_back(3);
  sizes.push_back(4);
  relu_input->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  relu_input->setSizes(sizes);
  relu_input->setUniqueName("relu_input");
  auto relu_node = relu_graph->create(Symbol("Relu"), relu_graph->inputs());
  relu_graph->appendNode(relu_node);
  auto relu_output = relu_node->output();
  relu_output->setElemType(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  relu_output->setSizes(sizes);
  relu_output->setUniqueName("relu_output");
  relu_graph->return_node()->addInput(relu_output);

  // Set up IO information
  uint64_t shape[3] = {2, 3, 4};
  onnxTensorDescriptorV1 output;
  onnxTensorDescriptorV1 input;
  output.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  output.name = "relu_output";
  output.dataType = ONNXIFI_DATATYPE_FLOAT32;
  output.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  output.dimensions = 3;
  output.shape = shape;
  output.buffer = (onnxPointer) new float[24];
  input.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  input.name = "relu_input";
  input.dataType = ONNXIFI_DATATYPE_FLOAT32;
  input.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  input.dimensions = 3;
  input.shape = shape;
  input.buffer = (onnxPointer) new float[24];

  // Setup events
  // Hacky event usage to make it work (cannot use onnxifi with backend)
  onnxMemoryFenceV1 inputFence;
  inputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
  inputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
  auto inputEvent = new EventControl();
  inputEvent->signalled_ = true;
  inputFence.event = reinterpret_cast<onnxEvent>(inputEvent);
  onnxMemoryFenceV1 outputFence;
  outputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
  outputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
  auto outputEvent = new EventControl();
  outputEvent->signalled_ = false;
  outputFence.event = reinterpret_cast<onnxEvent>(outputEvent);

  // Execute using XLA backend
  XlaTransform runner(NULL, std::move(relu_graph), "relu", 0, nullptr);
  runner.translateGraph();
  auto executor = runner.executor();
  executor->initIO(1, &input, 1, &output);
  float* input_ptr = (float*)input.buffer;
  std::uniform_real_distribution<float> unif(-0.5, 0.5);
  for (int i = 0; i < 24; ++i) {
    std::random_device rand_dev;
    std::mt19937 rand_engine(rand_dev());
    input_ptr[i] = unif(rand_engine);
  }
  executor->executeComputation(&inputFence, &outputFence);

  // Check correctness
  ONNX_ASSERT(outputEvent->signalled_);
  float* output_ptr = (float*)output.buffer;
  for (int i = 0; i < 24; ++i) {
    if (input_ptr[i] > 0.0f) {
      ONNX_ASSERT(almost_equal(input_ptr[i], output_ptr[i]));
    } else {
      ONNX_ASSERT(almost_equal(0.0f, output_ptr[i]));
    }
  }

  // Free memory
  delete executor;
  delete[] input_ptr;
  delete[] output_ptr;
  delete inputEvent;
  delete outputEvent;
}
}
