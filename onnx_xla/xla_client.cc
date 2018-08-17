#include "onnx_xla/xla_client.h"
#include "onnx/onnx.pb.h"
#include "onnx/proto_utils.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include <vector>

namespace onnx_xla {

XlaClient::XlaClient(const std::string& target) {
  auto channel =
      grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
  channel->WaitForConnected(gpr_time_add(
      gpr_now(GPR_CLOCK_REALTIME), gpr_time_from_seconds(10, GPR_TIMESPAN)));
  std::cout << "Channel to server is connected on target " << target
            << std::endl;
  xla_service_ = xla::grpc::XlaService::NewStub(channel);
}

std::string XlaClient::TryRun() {
  // build the computation graph
  xla::XlaBuilder builder("simple_add");
  const auto x_param = builder.Parameter(
      0, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2}), "x");
  const auto y_param = builder.Parameter(
      1, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2}), "y");
  builder.Add(x_param, y_param);
  const auto a_param = builder.Parameter(
      2, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2}), "a");
  const auto b_param = builder.Parameter(
      3, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2}), "b");
  std::vector<xla::XlaOp> retValues;
  retValues.push_back(builder.Add(x_param, y_param));
  retValues.push_back(builder.Add(a_param, b_param));
  builder.Tuple(retValues);
  auto computation = builder.Build().ConsumeValueOrDie();

  // feed the inputs
  auto x_literal = xla::Literal::CreateR1<float>({1., 2.});
  auto y_literal = xla::Literal::CreateR1<float>({3., 4.});
  auto a_literal = xla::Literal::CreateR1<float>({4., 5.});
  auto b_literal = xla::Literal::CreateR1<float>({6., 7.});

  auto x_data = xla::TransferParameterToServer(*x_literal.release());
  auto y_data = xla::TransferParameterToServer(*y_literal.release());
  auto a_data = xla::TransferParameterToServer(*a_literal.release());
  auto b_data = xla::TransferParameterToServer(*b_literal.release());
  // execute
  std::unique_ptr<xla::Literal> result_comp = xla::ExecuteComputation(
      computation,
      {x_data.release(), y_data.release(), a_data.release(), b_data.release()});
  //  std::cout << result_comp->DecomposeTuple().size() << std::endl;
  std::vector<xla::Literal> result_literals = result_comp->DecomposeTuple();
  // print result
  ONNX_NAMESPACE::TensorProto result;
  result.set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  for (const auto dim : result_literals[1].shape().dimensions()) {
    result.add_dims(dim);
  }
  for (const auto v : result_literals[1].data<float>()) {
    result.add_float_data(v);
  }
  return ONNX_NAMESPACE::ProtoDebugString(result);
}
}
