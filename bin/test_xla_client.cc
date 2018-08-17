//Test client written to communicate with server - will be removed.

#include "onnx_xla/xla_client.h"
#include <iostream>

int main(int argc, char **argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  onnx_xla::XlaClient client("localhost:51000");
  std::string reply = client.TryRun();
  std::cout << "Client received: " << reply << std::endl;

  return 0;
}
