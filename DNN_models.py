import simpy
from simpy.events import AnyOf, AllOf
from collections import deque
from core import Job

ij = 0

class Layer:
    """
    The building block of a DAG.
    It is a generic layer class that can describe any type of DNN layer.
    """
    def __init__(self, tensor_size, is_trainable, forward_pass_units=None, backward_pass_units=None,
                 input_layers=None, output_layers=None, forward_dependencies=None, backward_dependencies=None,
                 **extras):
        """
        :param tensor_size: The size of the tensor that is produced (Not the dimensions!).
        :param is_trainable: Whether this layer has trainable parameters or not. If it does not then it is only included
        for its computational cost (No communication cost).
        :param forward_pass_units: The processing cost of a forward pass. If none, then the tensor_size is used.
        :param backward_pass_units: The processing cost of a backward pass. If none, then tensor_size is used
        :param input_layers: A list of layers that provide the input to this layer.
        :param output_layers: A list of layers that will receive the output of this layer.
        :param forward_dependencies: A set of layers that this layer depends on in a forward pass.
        :param backward_dependencies: A set of layers that this layer depends on in a backward pass.
        :param extras: Custom attributes that help identify this layer or its behavior.
        """
        self.tensor_size = int(tensor_size)
        self.is_trainable = is_trainable
        self.tensor_size = tensor_size
        if forward_pass_units is None:
            self.forward_pass_units = self.tensor_size
        else:
            self.forward_pass_units = int(forward_pass_units)
        if backward_pass_units is None:
            self.backward_pass_units = self.tensor_size
        else:
            self.backward_pass_units = int(backward_pass_units)
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.extras = extras
        self.forward_dependencies = forward_dependencies
        self.backward_dependencies = backward_dependencies
        self.traversal_flag = False # Used to mark the layer in different traversal algorithms


class DNN:
    """
    Basically a DAG of layers.
    It is a generic class that can describe and simulate the behavior of any DNN architecture.
    """
    def __init__(self, dag_root):
        self.dag_root = dag_root
        self.linearized_dag = None

    def _linearize_dag(self):
        """
        A method that traverses the dag and produces the processing order of the layers.
        Uses a blocking breadth first search. Where a layer that does not have all of its input layers processed is,
        considered as a blocked path.
        (Implementation may be able to be more efficient)
        """
        linearized_dag = list()
        current_processing_list = [self.dag_root]
        next_processing_list = list()
        while len(current_processing_list) > 0:
            to_process = current_processing_list.pop(0)
            all_input_processed = True
            for dep in to_process.forward_dependencies:
                if not dep.traversal_flag:  # If we haven't processed this layer yet
                    all_input_processed = False
                    break
            if not all_input_processed:
                continue
            linearized_dag.append(to_process)
            if to_process.output_layers is not None:
                next_processing_list.extend(to_process.output_layers)
            to_process.traversal_flag = True
            if len(current_processing_list) == 0:
                current_processing_list = next_processing_list
                next_processing_list = list()

        self.linearized_dag = linearized_dag

    def _set_all_dependencies(self, recursive_layer=None):
        """
        Attaches a forward and backward dependency set for each layer.
        It adds the two sets to the extras dict in each layer.
        Current implementation is O(n^2). However, i suspect that it can be optimized and reduced to O(n).
        Nevertheless since this is a discrete time simulation and since layers are usually in the orders of tens or
        hundreds, we can get away with this.
        """
        global ij
        print("Recursive depth: {}".format(ij))
        ij += 1
        if recursive_layer is None:
            recursive_layer = self.dag_root
        print("Layer: {}".format(recursive_layer.extras['index']))
        if recursive_layer.input_layers is not None:
            print("Input")
            for layer in recursive_layer.input_layers:
                if layer.forward_dependencies is None:
                    layer.forward_dependencies = self._get_dependencies(layer, forward=True)
                    self._set_all_dependencies(recursive_layer=layer)
        if recursive_layer.output_layers is not None:
            print("Output")
            for layer in recursive_layer.output_layers:
                if layer.backward_dependencies is None:
                    layer.backward_dependencies = self._get_dependencies(layer, forward=False)
                    self._set_all_dependencies(recursive_layer=layer)

    def _get_dependencies(self, layer, forward=True, accumulated=None):
        """
        A recursive function that uses a depth first search to get all of the layer dependencies of this single layer.
        :param layer: The layer that you are querying about
        :param forward: Whether we want to move forward (Check all input layers) or backward (Check all output layers)
        :param accumulated: The set used to maintain all dependencies throughout the recursion
        :return: A set of all the dependencies for the layer provided
        """
        if accumulated is None:
            accumulated = set()
        if forward:
            if layer.input_layers is None:
                return accumulated
        else:
            if layer.output_layers is None:
                return accumulated
        for dep in layer.input_layers if forward else layer.output_layers:
            if dep in accumulated:
                continue
            accumulated.add(dep)
            self._get_dependencies(dep, forward=forward, accumulated=accumulated)
        return accumulated

    def train(self, env: simpy.Environment, n_of_batches, batch_size, computation_queue, communication_queue):
        """
        :param env: The simpy environment used in this simulation.
        :param n_of_batches: How many batches should we run
        :param batch_size: How many forward & backward passes before we need to update the parameters or synchronize.
        :param computation_queue: Used to queue computational jobs
        :param communication_queue: Used to queue communication jobs
        """
        pass

    def forward_pass(self, env: simpy.Environment, computation_queue, communication_queue, ongoing_layer_jobs=None):
        """
        :param env: The simpy environment used in this simulation.
        :param computation_queue: Used to queue computational jobs
        :param communication_queue: Used to queue communication jobs
        :param ongoing_layer_jobs: A dict with key: layer and value: event to wait for before being able to do a
        forward pass on that layer. If none then we simply do not wait.
        :return: A dict with key: layer and value: event to wait for before being able to do a backward pass on that layer
        """
        forward_pass_output = dict()
        for layer in self.linearized_dag:
            # Wait for all dependencies to finish
            if ongoing_layer_jobs is not None:
                for dep in layer.forward_dependencies:
                    yield ongoing_layer_jobs[dep]
            # Create job
            job_extras = layer.extras.copy()
            job_extras["type"] = "forward_pass"
            job = Job(env, layer.forward_pass_units, **job_extras)
            forward_pass_output[layer] = job
            # Queue job
            computation_queue.queue(job)
            yield job
        return forward_pass_output

    def backward_pass(self, env: simpy.Environment, computation_queue, communication_queue, ongoing_layer_jobs=None):
        """
        :param env: The simpy environment used in this simulation.
        :param computation_queue: Used to queue computational jobs
        :param communication_queue: Used to queue communication jobs
        :param ongoing_layer_jobs: A dict with key: layer and value: event to wait for before being able to do a
        backward pass on that layer. If none then we simply do not wait.
        :return: A dict with key: layer and value: event to wait for before being able to do a forward pass on that
        layer
        """
        backward_pass_output = dict()
        for layer in self.linearized_dag:
            # Wait for all dependencies to finish
            if ongoing_layer_jobs is not None:
                for dep in layer.backward_dependencies:
                    yield ongoing_layer_jobs[dep]
            # Create job
            job_extras = layer.extras.copy()
            job_extras["type"] = "backward_pass"
            comp_job = Job(env, layer.forward_pass_units, **job_extras)
            job_extras = layer.extras.copy()
            job_extras["type"] = "parameter_communication"
            comm_job = Job(env, layer.forward_pass_units, **job_extras)
            backward_pass_output[layer] = comm_job
            # We only wait for the computational job.
            yield computation_queue.queue(comp_job)
            communication_queue.queue(comm_job)
        return backward_pass_output


class HomogeneousLayerDNN(DNN):
    """
    A DNN that consists of a linear graph of identical layers which all contain trainable parameters.
    Used for quick verification.
    """
    def __init__(self, n_of_layers, layer_size):
        root = Layer(layer_size, is_trainable=True, index=0)
        prev = root
        for i in range(1, n_of_layers):
            new = Layer(layer_size, is_trainable=True, input_layers=[prev], index=i)
            prev.output_layers = [new]
            prev = new
        super().__init__(root)


def xml_to_dnn(path_to_file):
    pass


def dnn_to_xml(path_to_file):
    pass


if __name__ == "__main__":
    """
    An example usage
    """
    model = HomogeneousLayerDNN(5, 4)
    model._set_all_dependencies()
    node = [model.dag_root]
    while node is not None:
        print("i: {} for_dep: {}".format(node[0].extras['index'],
                                     [x.extras['index'] for x in node[0].forward_dependencies]))
        node = node[0].output_layers
