#pragma once
// Test client - to be removed
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <string>

namespace onnx_xla {
class XlaClient {
 public:
  XlaClient(const std::string& target);
  std::string TryRun();

 private:
  std::unique_ptr<xla::grpc::XlaService::Stub> xla_service_{nullptr};
};
}
