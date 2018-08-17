from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import onnx.backend.test

from onnxifi_backend import OnnxifiBackend, OnnxifiBackendRep

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(OnnxifiBackend(), __name__)

backend_test.include(r'(test_relu'  # Test relu.
                     '|test_softmax' # Test softmax
                     '|test_batchnorm' # Test BatchNormalization
                     '|test_gemm' # Test Gemm
                     '|test_concat' # Test Concat
                     '|test_unsqueeze' # Test Unsqueeze
                     '|test_globalaveragepool' #Test GlobalAveragePool
                     '|test_lrn' # Test LRN
                     '|test_add' # Test Add
                     '|test_sub' # Test Sub
                     '|test_div' # Test Div
                     '|test_mul' # Test Mul
                     '|test_sum' # Test Sum
                     '|test_transpose' #Test Transpose
                     '|test_maxpool' # Test MaxPool
                     '|test_basic_conv' # Test Conv
                     '|test_conv_with'
                     '|test_average_pool' #Test AveragePool
                     '|test_dropout' # Test Dropout
                     '|test_resnet50' 
                     '|test_bvlc_alexnet'
                     '|test_densenet121'
                     '|test_inception_v1'
                     '|test_inception_v2'
#                     '|test_shufflenet'
                     '|test_squeezenet'
                     '|test_vgg19'
                     '|test_zfnet512'
                     ')')

backend_test.exclude(r'(test_maxpool_with_argmax' # MaxPool with indices not yet supported
                     ')')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
