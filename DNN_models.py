import simpy
from collections import deque


class Tensor:
    def __init__(self, size, computation_units=None, layer=None, name=None):
        if computation_units:
            self.computation_units = computation_units
        else:
            self.computation_units = size
        self.size = size
        self.layer = layer
        self.name = name


class Layer:
    def __init__(self, tensor_definitions, name=None):
        self.tensor_definitions = tensor_definitions
        self.name = name
        self.forward_passes = 0
        self.backward_passes = 0

    def forward_pass(self):
        self.forward_passes += 1

    def backward_pass(self):
        self.backward_passes += 1


class HomogeneousLayerModel:
    """Simulates a DNN model that consists of a linear graph of equally sized layers"""
    def __init__(self, env: simpy.Environment, num_layers=5, layer_size=10, layer_computation_units=None):
        self.layers = [Layer(layer_size, layer_computation_units, i) for i in range(num_layers)]
        self.env = env

    def train(self, processor, scheduler=None, batches=1):
        print("Starting {} train process".format(type(self).__name__))
        for batch in range(batches):
            # Forward pass
            # Consume received gradients here (Not yet sending anything back from the aggregator)

            # Backpropagation
            for layer in reversed(self.layers):
                tensor = yield self.env.process(self._compute_tensor(layer, processor))
                scheduler.schedule(tensor)

    def _produce_gradient_tensors(self, layer: Layer, processor: Processor):
        yield self.env.process(processor.queue(layer.computation_units))
        layer.tensor_count += 1
        return Tensor(layer.size, layer, layer.tensor_count)
