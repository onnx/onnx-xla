#include "onnx_xla/backend_test.h"
#include <iostream>

//Run ./tests from the build/ directory to run the executable.
//Pair of tests that run an IR graph with a relu operator on 
//the XLA backend. The static version passes the input value
//as an initializer, whereas the dynamic version passes the input
//value as an input.
int main(int argc, char **argv) {
  onnx_xla::static_relu_test();
  std::cout << "static_relu_test succeeded!" << std::endl;
  onnx_xla::dynamic_relu_test();
  std::cout << "dynamic_relu_test succeeded!" << std::endl;

  return 0;
}
