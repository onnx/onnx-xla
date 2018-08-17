#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <sys/stat.h> 
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "onnx/common/assertions.h"
#include "onnx/onnxifi.h"

//Basic test case that runs an onnx model of a graph with the relu operator
//(from the onnx repository) on the XLA backend. The ONNXIFI interface is used
//to run a ModelProto on the XLA backend.

using namespace ONNX_NAMESPACE;

bool almost_equal(float a, float b, float epsilon = 1e-5)  {
  return std::abs(a - b) < epsilon;
}

int main(int argc, char** argv)  {
  
  //Initialize backend
  onnxBackendID backendIDs;
  size_t numBackends = 1;
  onnxBackend backend;
  if (onnxGetBackendIDs(&backendIDs, &numBackends) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error getting backend IDs" << std::endl;
  }
  if (onnxInitBackend(backendIDs, nullptr, &backend) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error initializing backend" << std::endl;
  }

  //Read in model from file
  const void* buffer;
  int size = 0;
  int fd = ::open("../third_party/onnx/onnx/examples/resources/single_relu.onnx", O_RDONLY);
  google::protobuf::io::FileInputStream raw_input(fd);
  raw_input.SetCloseOnDelete(true);
  google::protobuf::io::CodedInputStream coded_input(&raw_input);
  coded_input.GetDirectBufferPointer(&buffer, &size);

 //Fill in I/O information
  uint64_t shape[2] = {1, 2};

  uint32_t inputsCount = 1;
  onnxTensorDescriptorV1 input;
  input.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  input.name = "x";
  input.dataType = ONNXIFI_DATATYPE_FLOAT32;
  input.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  input.dimensions = 2;
  input.shape = shape;
  input.buffer = (onnxPointer) new float[24];

  uint32_t outputsCount = 1;
  onnxTensorDescriptorV1 output;  
  output.tag = ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1;
  output.name = "y";
  output.dataType = ONNXIFI_DATATYPE_FLOAT32;
  output.memoryType = ONNXIFI_MEMORY_TYPE_CPU;
  output.dimensions = 2;
  output.shape = shape;
  output.buffer = (onnxPointer) new float[24];

  float* input_ptr = (float*) input.buffer;
  std::uniform_real_distribution<float> unif(-0.5, 0.5);
  for (int i = 0; i < 2; ++i)  {
    std::random_device rand_dev;
    std::mt19937 rand_engine(rand_dev());
    input_ptr[i] = unif(rand_engine);
  }
  float *output_ptr = (float*) output.buffer;

  //Set up I/O memory fences
  onnxMemoryFenceV1 inputFence;
  inputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
  inputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
  if (onnxInitEvent(backend, &inputFence.event) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error initializing event for input memory fence" << std::endl;
  }
  onnxMemoryFenceV1 outputFence;
  outputFence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
  outputFence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
 
 //Run the graph
  onnxGraph graph;
  if (onnxInitGraph(backend, NULL, (size_t) size, buffer, 0,
                nullptr, &graph) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error initializing graph" << std::endl;
  }
  if (onnxSetGraphIO(graph, inputsCount, &input,
                 outputsCount, &output) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error setting Graph IO" << std::endl;
  }
  if (onnxSignalEvent(inputFence.event) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error signalling event for input memory fence" << std::endl;
  }
  if (onnxRunGraph(graph, &inputFence, &outputFence) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error running Graph" << std::endl;
  }

  //Check correctness
  if (onnxWaitEvent(outputFence.event) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error waiting for event for output fence" << std:: endl;
  }

  for (int i = 0; i < 2; ++i)  {
    if (input_ptr[i] > 0.0f)  {
     ONNX_ASSERT(almost_equal(input_ptr[i], output_ptr[i]));
    } else {
     ONNX_ASSERT(almost_equal(0.0f, output_ptr[i]));
    }
  }
  delete [] input_ptr;
  delete [] output_ptr;
  
  //Release graph and backend resources
  if (onnxReleaseGraph(graph) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error releasing graph" << std::endl;
  }
  if (onnxReleaseEvent(inputFence.event) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Erro releasing event for input fence" << std::endl;
  }
  if (onnxReleaseEvent(outputFence.event) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Erro releasing event for output fence" << std::endl;
  }
  if (onnxReleaseBackend(backend) != ONNXIFI_STATUS_SUCCESS)  {
    std::cerr << "Error releasing backend" << std::endl;
  }
  if (onnxReleaseBackendID(backendIDs) != ONNXIFI_STATUS_SUCCESS)  {  
    std::cerr << "Error releasing backendID" << std::endl;
  }

  return 0;
}


