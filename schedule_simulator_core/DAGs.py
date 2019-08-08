"""
This module includes everything related to DAGs and their algorithms
Building the architecture is done here. Using the DAG architecture as a DNN is done in the DNN_functions module
"""

# Extra keys that have this prefix should not be propagated to sub objects created.
LOCAL_EXTRA_PREFIX = "$local$"


class Layer:
    """
    The building block of a DAG.
    It is a generic layer class that can describe any type of DNN layer.
    """
    def __init__(self, forward_pass_units, backward_pass_units, communication_units,
                 input_layers=None, output_layers=None, **extras):
        """
        :param forward_pass_units: The processing cost of a forward pass.
        :param backward_pass_units: The processing cost of a backward pass.
        :param communication_units: The communication cost of a backward pass.
        :param input_layers: A list of layers that provide the input to this layer.
        :param output_layers: A list of layers that will receive the output of this layer.
        :param forward_dependencies: A set of layers that this layer depends on in a forward pass.
        :param backward_dependencies: A set of layers that this layer depends on in a backward pass.
        :param extras: Custom attributes that help identify this layer or its behavior.
        """
        self.forward_pass_units = forward_pass_units
        self.backward_pass_units = backward_pass_units
        self.communication_units = communication_units
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.extras = extras
        self._forward_dependencies = None
        self._backward_dependencies = None

    def __str__(self):
        return str(self.extras)


class LayerFactory:
    """
    A class that is used to generate layers. Mainly used with distributions to allow for some randomness when building
    a dag.
    """
    def __init__(self, forward_pass_units, backward_pass_units, communication_units, indexing_offset=0, **extras):
        """
        :param forward_pass_units: Constant or distribution
        :param backward_pass_units: Constant or distribution
        :param communication_units: Constant or distribution
        :param indexing_offset: Layers will be given incremental indices in their extras field. Should we offset the
        starting index ?
        :param extras: Any extras that will be passed on when creating the layer and later passed on to all created jobs
        """
        self.forward_pass_units = forward_pass_units
        self.backward_pass_units = backward_pass_units
        self.communication_units = communication_units
        self.indexing_offset = indexing_offset
        self.extras = extras
        self.count = 0

    def create_layer(self):
        attributes = list()
        for attr in [self.forward_pass_units, self.backward_pass_units, self.communication_units]:
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
    def __init__(self, dag_input_layers, dag_output_layers=None, **extras):
        self.dag_input_layers = dag_input_layers
        self.dag_output_layers = list()
        self.extras = extras
        # Extract dag_output_layers
        if dag_output_layers is None:
            self.set_output_layers()
        # Extract layer dependencies
        self.extract_dependencies()
        # Extract topological order
        self.topological_order = None
        self.produce_topological_order()

    def remove_layer(self, layer):
        for inp in layer.input_layers:
            inp.output_layers.remove(layer)
            inp.output_layers.extend(layer.output_layers)
        for out in layer.output_layers:
            out.input_layers.remove(layer)
            out.input_layers.extend(layer.input_layers)
        if layer in self.dag_input_layers:
            self.dag_input_layers.remove(layer)
            self.dag_input_layers.extend(layer.output_layers)
        if layer in self.dag_output_layers:
            self.dag_output_layers.remove(layer)
            self.dag_output_layers.extend(layer.input_layers)

    def set_output_layers(self):
        def process_node(node):
            if node.output_layers is None:
                self.dag_output_layers.append(node)
        self.traverse_BFS(processing_function=process_node)

    def produce_topological_order(self):
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
            traverse(root=root, visited=visited)

    def extract_dependencies(self):
        """
        Uses a DFS O(n) to extract all dependencies between layers in the DAG.
        I should modify this later to use the generic DFS post order function defined above.
        """
        # Reset dependencies
        def process_node(node):
            node._forward_dependencies = set()
            node._backward_dependencies = set()
        self.traverse_BFS(processing_function=process_node)

        def extract_forward_dependencies(root: Layer, deps: set, visited: set):
            if root in visited:
                return
            root._forward_dependencies = root._forward_dependencies.union(deps)
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
            root._backward_dependencies = root._backward_dependencies.union(deps)
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

    def __copy__(self):
        """
        TODO Provide a concrete cloning method instead
        """
        return deserialize_dag(serialize_dag(self))

    def copy(self):
        return self.__copy__()

    def get_layer_costs(self):
        fp_units = list()
        bp_units = list()
        comm_units = list()
        comp_units = list()
        for layer in self.topological_order:
            fp_units.append(layer.forward_pass_units)
            bp_units.append(layer.backward_pass_units)
            comm_units.append(layer.communication_units)
            comp_units.append(layer.forward_pass_units+layer.backward_pass_units)
        return dict(fp_units=fp_units, bp_units=bp_units, comm_units=comm_units, comp_units=comp_units)

    def __str__(self):
        s = ""
        for key, value in self.extras.items():
            if key.startswith(LOCAL_EXTRA_PREFIX):
                continue
            s += "{}:{} ".format(key, value)
        return s

    # TODO Should add some functions to make it easy to join different DAGs together


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
            new.input_layers = [prev]
            prev.output_layers = [new]
            prev = new
        super().__init__([root])


class HomogeneousLinearDAG(LinearDag):
    """
    A DAG that consists of a linear graph of identical layers which all contain trainable parameters.
    Used for quick verification.
    """
    def __init__(self, n_of_layers, fp_units, bp_units, comm_units, is_trainable=True, indexing_offset=0):
        layer_factory = LayerFactory(fp_units, bp_units, comm_units, indexing_offset=indexing_offset)
        super().__init__(n_of_layers, layer_factory)


class RandomDAG(DAG):
    """
    A DAG that consists of layers in a semi random configuration governed by a linearity coefficient.
    Layers are built using a layer_factory
    """
    def __init__(self, n_of_layers, layer_factory: LayerFactory, linearity_coefficient=0.5):
        pass


def serialize_dag(dag: DAG, formatted=True):
    """
    A function that returns a json string that is simply a list of layers. With each layer's connections described.
    """
    import json
    i = 0
    temp_ids = dict()
    serialized_dag = {"extras": dag.extras, "layers": dict()}

    def add_layer(layer):
        nonlocal i
        sl = dict()
        if layer not in temp_ids:
            temp_ids[layer] = i
            i += 1
        sl["forward_pass_units"] = layer.forward_pass_units
        sl["backward_pass_units"] = layer.backward_pass_units
        sl["communication_units"] = layer.communication_units
        sl["input_layers"] = list()
        if layer.input_layers is not None:
            for input_layer in layer.input_layers:
                if input_layer not in temp_ids:
                    temp_ids[input_layer] = i
                    i += 1
                sl["input_layers"].append(temp_ids[input_layer])
        sl["output_layers"] = list()
        if layer.output_layers is not None:
            for output_layer in layer.output_layers:
                if output_layer not in temp_ids:
                    temp_ids[output_layer] = i
                    i += 1
                sl["output_layers"].append(temp_ids[output_layer])
        sl["extras"] = layer.extras
        serialized_dag["layers"][temp_ids[layer]] = sl
    dag.traverse_BFS(add_layer)
    return json.dumps(serialized_dag, indent=4) if formatted else json.dumps(serialized_dag)


def deserialize_dag(serialized_dag):
    import json
    serialized_dag = json.loads(serialized_dag)
    temp_ids = dict()
    input_layers = list()
    for i, layer_dict in serialized_dag["layers"].items():
        extras = layer_dict["extras"]
        del(layer_dict["extras"])
        temp_ids[i] = Layer(**layer_dict, **extras)
    # Translate layer ids to objects in layer lists
    for layer in temp_ids.values():
        object_input_layers = list()
        object_output_layers = list()
        if len(layer.input_layers) == 0:
            input_layers.append(layer)
        else:
            for input_layer_i in layer.input_layers:
                object_input_layers.append(temp_ids[str(input_layer_i)])
        if len(layer.output_layers) > 0:
            for output_layer_i in layer.output_layers:
                object_output_layers.append(temp_ids[str(output_layer_i)])
        layer.input_layers = object_input_layers
        layer.output_layers = object_output_layers
    dag = DAG(input_layers, **serialized_dag["extras"])
    return dag


if __name__ == "__main__":
    """
    An example usage
    """
    dag = HomogeneousLinearDAG(5, 4, 4, 4)
    print(serialize_dag(dag))
    def p(node):
        print(node.extras['index'])
    dag.traverse_DFS(p, order="pre-order")
