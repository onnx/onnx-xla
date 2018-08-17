from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from python_onnxifi import Backend, BackendRep
import onnx
from onnx import (NodeProto,
                  ModelProto)

class OnnxifiBackendRep(object):
    def __init__(self, backendRep):
        self.backend_rep_ = backendRep

    # Inputs is a list of numpy arrays corresponding to ModelProto
    def run(self, inputs, **kwargs):
        return self.backend_rep_.run(inputs, **kwargs)


class OnnxifiBackend(object):
    def __init__(self):
        self.backend_ = Backend()

    #Use ONNXIFI interface to determine compatibility of model
    def is_compatible(self,
                      model,  # type: ModelProto
                      device='CPU',  # type: Text
                      **kwargs  # type: Any
                      ):  # type: (...) -> bool
        return self.backend_.is_compatible(model.SerializeToString(), device, **kwargs)

    # Sets up model on given backend device
    # Returns OnnxifiBackendRep object to be run
    # TODO: Use kwargs to pass in weightDescriptors
    def prepare(self,
                model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> BackendRep
        onnx.checker.check_model(model)
        return OnnxifiBackendRep(self.backend_.prepare(model.SerializeToString(), device, **kwargs))

    # Runs model using list of numpy inputs
    # Returns list of outputs
    def run_model(self,
                  model,  # type: ModelProto
                  inputs,  # type: Any
                  device='CPU',  # type: Text
                  **kwargs  # type: Any
                  ):  # type: (...) -> Tuple[Any, ...]
        backendRep = self.prepare(model, device, **kwargs)
        assert backendRep is not None
        return backendRep.run(inputs, **kwargs)

    #TODO: Implement run_node
    def run_node(cls,
                 node,  # type: NodeProto
                 inputs,  # type: Any
                 device='CPU',  # type: Text
                 outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
                 **kwargs  # type: Dict[Text, Any]
                 ):  # type: (...) -> Optional[Tuple[Any, ...]]
        return None

    # Returns boolean indicating whether the backend with given device is supported
    def supports_device(self, device):  # type: (Text) -> bool
        return self.backend_.supports_device(device)

    # Utility function return information about available devices
    def get_devices_info(self): #type: () -> [Sequence[Tuple[string, string]]]
        return self.backend_.get_devices_info()

