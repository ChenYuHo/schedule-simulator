import torch
import torchvision.models as models
from schedule_simulator_core.DAGs import Layer, DAG, LOCAL_EXTRA_PREFIX
from model_extraction.pytorch_utils import *
import torch.jit as jit


def pytorch_model_to_DAG(model):
    def add_layer(layer):
        pass


def extract_costs_from_profile():
    pass


def apply_layer_costs_to_dag():
    pass


def parse_jit_trace(trace):
    nodes = dict()
    graph_lines = str(trace.graph).split("\n")
    for line in graph_lines:
        child = list()
        parent = list()
        parents = list()
        recording_child = True
        recording_word = False
        for c in line:
            if c == "%":
                recording_word = True
            elif recording_word:
                if c != " ":
                    if recording_child:
                        child.append(c)
                    else:
                        parent.append(c)
                else:
                    recording_word = False
                    if recording_child:
                        child = "".join(child)
                        recording_child = False
                    else:
                        parents.append("".join(parent))
                        parent.clear()
        nodes[child] = parents
    print(nodes)


net = models.inception_v3()
inputs, _ = get_dummy_input_output(net, batch_size=2, device="cpu")
trace = jit.trace(net, inputs, check_trace=False)
print(trace.graph)
parse_jit_trace(trace)
