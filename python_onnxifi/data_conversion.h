#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include "onnx/onnxifi.h"
#include "onnx/onnx.pb.h"
#include "onnx/shape_inference/implementation.h"

#include <vector>
#include <unordered_map>

namespace py = pybind11;

using ::ONNX_NAMESPACE::ModelProto;
using ::ONNX_NAMESPACE::ValueInfoProto;

// Utility classes to navigate conversions of onnxTensorDescriptorV1 and numpy

// Struct to manage onnxTensorDescriptorV1 data for lifetime of the descriptor
struct DescriptorData {
  // Setup metadata and allocate buffer
  DescriptorData(const ValueInfoProto& vip);
  // Ensure pointers point to beginning of vectors/strings
  DescriptorData(const DescriptorData& d);
  DescriptorData(DescriptorData&& d) noexcept;

  onnxTensorDescriptorV1 descriptor;
  // Containers for memory managment
  std::vector<uint64_t> shape;
  std::string name;
  std::vector<char> buffer;

  // Returns size of buffer need to store data of onnxType with given shape
  template <typename onnx_type, typename unused>
  static void getBufferSize(uint64_t& buffer_size,
                            const uint64_t* shape,
                            uint32_t dimensions);
};

// TODO: Weight descriptor conversions
// TODO: Support for strides
class DataConversion final {
 public:
  // Constructor creates DataConversion object with underlying model
  // Initializes member variables with descriptor metadata and allocates buffers
  // Throws if input or output descriptor data was not found/inferred
  // 1 DataConversion object for every onnxGraph or BackendRep
  DataConversion(const std::string& serializedModel);

  // Returns list of numpy outputs from descriptor buffers in order of
  // ModelProto outputs
  py::list getOutputs() const;

  // Sets inputs in tensor descriptor buffers from numpy arrays
  void setInputs(const py::list& inputs);

  // Return tensor descriptors from the stored member variables
  std::vector<onnxTensorDescriptorV1> getInputDescriptors() const;
  std::vector<onnxTensorDescriptorV1> getOutputDescriptors() const;

 private:
  /************************************************************************/
  /********  DESCRIPTOR DATA    --->   NUMPY ARRAYS   *********************/

  // Adds array corresponding to dd to numpyArrayList
  template <typename onnx_type, typename py_type>
  static void addNumpyArray(py::list& numpyArrayList, const DescriptorData& dd);

  // Appends to numpyArrayList with arrays corresponding to descriptorsData
  static void fillNumpyArrayList(
      py::list& numpyArrayList,
      const std::vector<DescriptorData>& descriptorsData);

  /************************************************************************/
  /******** NUMPY ARRAYS    --->   DESCRIPTOR DATA   **********************/

  // Fills dd buffer with buffer data from arrayInfo
  // Throws if metadata does not match
  // Requires buffer to be allocated from before
  template <typename onnx_type, typename py_type>
  static void fillDescriptorDataImpl(
      DescriptorData& dd,
      const py::array& numpyArray,
      ONNX_NAMESPACE::TensorProto_DataType dataType);

  // Fills descriptorsData buffers with values from numpyArrayList
  // Metadata of descriptorsData must be filled and buffer must be allocated
  // Throws if numpy inputs and descriptors do not match or are unexpected
  static void fillDescriptorDataVector(
      const py::list& numpyArrayList,
      std::vector<DescriptorData>& descriptorsData);

  /************************************************************************/
  /********************     ADDITIONAL HELPERS  ***************************/

  // Returns vector of onnxTensorDescriptorV1 based on descriptorsData vector
  static std::vector<onnxTensorDescriptorV1> getTensorDescriptors(
      const std::vector<DescriptorData>& descriptorsData);

  /************************************************************************/
  /*********************PRIVATE MEMBER VARIABLES***************************/

  // Vectors of input, output, and weight tensor Descriptors
  std::vector<DescriptorData> input_descriptors_data_;
  std::vector<DescriptorData> output_descriptors_data_;

  // ModelProto model the object uses
  ModelProto model_;
};
