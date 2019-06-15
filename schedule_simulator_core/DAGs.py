"""
This module includes everything related to DAGs and their algorithms
Building the architecture is done here. Using the DAG architecture as a DNN is done in the DNN_functions module
"""


class Layer:
    """
    The building block of a DAG.
    It is a generic layer class that can describe any type of DNN layer.
    """
    def __init__(self, tensor_size, is_trainable, forward_pass_units=None, backward_pass_units=None,
                 communication_units=None,input_layers=None, output_layers=None, forward_dependencies=None,
                 backward_dependencies=None,
                 **extras):
        """
        :param tensor_size: The size of the tensor that is produced (Not the dimensions!).
        :param is_trainable: Whether this layer has trainable parameters or not. If it does not then it is only included
        for its computational cost (No communication cost).
        :param forward_pass_units: The processing cost of a forward pass. If none, then the tensor_size is used.
        :param backward_pass_units: The processing cost of a backward pass. If none, then tensor_size is used
        :param communication_units: The communication cost of a backward pass. If none, then tensor_size is used. Used
        to accommodate communication overheads.
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
        if communication_units is None:
            self.communication_units = self.tensor_size
        else:
            self.communication_units = int(communication_units)
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.extras = extras
        self.forward_dependencies = forward_dependencies
        self.backward_dependencies = backward_dependencies


class LayerFactory:
    """
    A class that is used to generate layers. Mainly used with distributions to allow for some randomness when building
    a dag.
    """
    def __init__(self, layer_size, forward_pass_units=None, backward_pass_units=None, communication_units=None,
                 is_trainable=True, indexing_offset=0, **extras):
        self.layer_size = layer_size
        self.forward_pass_units = forward_pass_units
        self.backward_pass_units = backward_pass_units
        self.communication_units = communication_units
        self.is_trainable = is_trainable
        self.indexing_offset = indexing_offset
        self.extras = extras
        self.count = 0

    def create_layer(self):
        attributes = list()
        for attr in [self.layer_size, self.is_trainable, self.forward_pass_units, self.communication_units,
                     self.communication_units]:
            try:
                attributes.append(attr.generate_value())
            except AttributeError:
                attributes.append(attr)
        extras = self.extras.copy()
        extras['index'] = self.count + self.indexing_offset
        self.count += 1
        return Layer(*attributes, **extras)


class DAG:
    """
    It is a generic class that can describe any DNN architecture.
    """
    def __init__(self, dag_input_layers):
        self.dag_input_layers = dag_input_layers
        self.dag_output_layers = list()
        # Traverse dag and setup layer variables as well as extract dag_output_layers
        def process_node(node):
            node.forward_dependencies = set()
            node.backward_dependencies = set()
            if node.output_layers is None:
                self.dag_output_layers.append(node)
        self.traverse_BFS(processing_function=process_node)
        # Extract layer dependencies
        self.extract_dependencies()
        # Extract topological order
        self.topological_order = list()
        def add_node(node):
            self.topological_order.append(node)
        self.traverse_DFS(add_node, order="post-order")
        self.topological_order.reverse()

    def traverse_BFS(self, processing_function):
        """
        Uses an iterative BFS O(n) to process all nodes in the dag using the processing_function.
        :param processing_function: The function that will be called on each node. should only require one argument
        which is the node being processed.
        """
        visited = set()
        current = self.dag_input_layers.copy()
        new = list()
        while len(current) > 0:
            node = current.pop(0)
            visited.add(node)
            if node.output_layers is not None:
                for out in node.output_layers:
                    if out not in visited:
                        new.append(out)
            processing_function(node)
            if len(current) == 0:
                current = new
                new = list()

    def traverse_DFS(self, processing_function, order="post-order"):
        """
        Uses a recursive DFS O(n) to process all nodes in the dag using the processing_function.
        :param processing_function: The function that will be called on each node. should only require one argument
        which is the node being processed.
        :param order: The order of traversal. "pre-order" or "post-order"
        """
        if order != "pre-order" and order != "post-order":
            raise Exception("Invalid order '{}' provided.".format(order))

        def traverse(root: Layer, visited: set):
            visited.add(root)
            if order == "pre-order":
                processing_function(root)
            if root.output_layers is not None:
                for child in root.output_layers:
                    if child not in visited:
                        traverse(root=child, visited=visited)
            if order == "post-order":
                processing_function(root)
        visited = set()
        for root in self.dag_input_layers:
            traverse(root=root,visited=visited)

    def extract_dependencies(self):
        """
        Uses a DFS O(n) to extract all dependencies between layers in the DAG.
        I should modify this later to use the generic DFS post order function defined above.
        """
        def extract_forward_dependencies(root: Layer, deps: set, visited: set):
            if root in visited:
                return
            root.forward_dependencies = root.forward_dependencies.union(deps)
            # We should traverse the node only if the node's ingoing connections have all been traversed.
            if root.input_layers is not None:
                for ingoing in root.input_layers:
                    if ingoing not in visited:
                        return
            # Process the node
            visited.add(root)
            deps.add(root)
            # Recurse using children
            if root.output_layers is not None:
                for child in root.output_layers:
                    extract_forward_dependencies(root=child, deps=deps, visited=visited)
            deps.remove(root)

        def extract_backward_dependencies(root: Layer, deps: set, visited: set):
            if root in visited:
                return
            root.backward_dependencies = root.backward_dependencies.union(deps)
            # We should traverse the node only if the node's outgoing connections have all been traversed.
            if root.output_layers is not None:
                for outgoing in root.output_layers:
                    if outgoing not in visited:
                        return
            # Process the node
            visited.add(root)
            deps.add(root)
            # Recurse using children
            if root.input_layers is not None:
                for child in root.input_layers:
                    extract_backward_dependencies(root=child, deps=deps, visited=visited)
            deps.remove(root)

        visited = set()
        for root in self.dag_input_layers:
            extract_forward_dependencies(root=root, deps=set(), visited=visited)
        visited = set()
        for root in self.dag_output_layers:
            extract_backward_dependencies(root=root, deps=set(), visited=visited)

    """Should add some functions to make it easy to join different DAGs together"""


class LinearDag(DAG):
    """
    A general linear DAG built using a layer factory
    """
    def __init__(self, n_of_layers, layer_factory: LayerFactory):
        self.layer_factory = layer_factory
        root = self.layer_factory.create_layer()
        prev = root
        for i in range(n_of_layers):
            new = self.layer_factory.create_layer()
            prev.output_layers = [new]
            prev = new
        super().__init__([root])


class HomogeneousLinearDAG(LinearDag):
    """
    A DAG that consists of a linear graph of identical layers which all contain trainable parameters.
    Used for quick verification.
    """
    def __init__(self, n_of_layers, layer_size, is_trainable=True, indexing_offset=0):
        layer_factory = LayerFactory(layer_size=layer_size, is_trainable=is_trainable, indexing_offset=indexing_offset)
        super().__init__(n_of_layers, layer_factory)


class RandomDAG(DAG):
    """
    A DAG that consists of layers in a semi random configuration governed by a linearity coefficient
    """
    def __init__(self, n_of_layers, layer_factory: LayerFactory, linearity_coefficient=0.5):
        pass


def serialize_dag(path_to_file):
    pass


def deserialize_dag(path_to_file):
    pass


if __name__ == "__main__":
    """
    An example usage
    """
    DAG = HomogeneousLinearDAG(5, 4)
    def p(node):
        print(node.extras['index'])
    DAG.traverse_DFS(p, order="pre-order")
