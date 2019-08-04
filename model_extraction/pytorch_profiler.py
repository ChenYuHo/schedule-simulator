import argparse
from datetime import datetime
import sys
import time
import json
import socket
from contextlib import nullcontext
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
sys.path.append("..")
from model_extraction.pytorch_utils import *


def profile(model, loss_func, optimizer, batch_size, num_of_batches, device="gpu", enable_autograd_profiler=False,
            verbosity=1):
    # Setup and inject timings hooks
    if device == "gpu":
        if not torch.cuda.is_available():
            raise Exception("No GPUs were detected. Change device to 'cpu' or make sure you have appropriate gpu "
                            "version of pytorch as well as the CUDA toolkit.")
        model = model.cuda()
        loss_func = loss_func.cuda()

    def init(module, input):
        start_times.append(time.time_ns())

    def get_hook(index, type):
        def hook(module, input, output):
            layer_time_stamps[index]["{}_pass_ts".format(type)].append(time.time_ns())
        return hook

    start_times = list()
    layer_time_stamps = list()
    layers = list()

    def register_hooks(module):
        module.register_forward_hook(get_hook(len(layers), "forward"))
        module.register_backward_hook(get_hook(len(layers), "backward"))
        layers.append(dict(type=type(module).__name__, forward_pass_cost=list(), backward_pass_cost=list()))
        layer_time_stamps.append(dict(forward_pass_ts=list(), backward_pass_ts=list()))

    model.register_forward_pre_hook(init)
    traverse_module(model, register_hooks)
    context = torch.autograd.profiler.profile(use_cuda=device == "gpu") if enable_autograd_profiler else nullcontext()
    # Run
    with context:
        train(model=model, loss_func=loss_func, optimizer=optimizer, batch_size=batch_size,
              num_of_batches=num_of_batches, device=device, verbosity=verbosity)
    # Parse costs from time stamps
    for batch_i in range(len(layer_time_stamps[0].values().__iter__().__next__())):
        last_ts = start_times[batch_i]
        for t in ["forward_pass", "backward_pass"]:
            for layer_i in range(len(layers)):
                if t == "backward_pass":
                    layer_i = len(layers) - layer_i - 1
                next_ts = layer_time_stamps[layer_i]["{}_ts".format(t)][batch_i]
                layers[layer_i]["{}_cost".format(t)].append(next_ts-last_ts)
                last_ts = next_ts
    # Name layers
    layer_costs = dict()
    for layer_i, layer_dict in enumerate(layers):
        layer_costs["{}_{}".format(layer_i, layer_dict["type"])] = dict(forward_pass_units=layer_dict["forward_pass_cost"],
                                                                        backward_pass_units=layer_dict["backward_pass_cost"])
    return layer_costs, context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that profiles pytorch models and produces layer wise timings.")
    parser.add_argument("model",
                        help="The torchvision model to profile. See TORCHVISION.MODELS documentation for all options.")
    parser.add_argument("-l", "--loss", default="CrossEntropyLoss",
                        help="The loss function to use. See TORCH.NN.MODULES.LOSS documentation for all options.")
    parser.add_argument("-o", "--optimizer", default="SGD",
                        help="The optimizer to use when training. See TORCH.OPTIM documentation for all options.")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="The device to use for all operations.")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="The batch size used for all functions")
    parser.add_argument("-nb", "--num_of_batches", type=int, default=8,
                        help="The number of batches to run")
    parser.add_argument("--out", help="File name to write the final report to")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="0: Suppress all output\n"
                             "1: Show profiler output\n")
    parser.add_argument("-st", "--save_trace", default=False, action="store_true",
                        help="Whether we save the trace of the training or not. For debugging.")
    args = parser.parse_args()

    def try_import(name, path):
        try:
            module = __import__(path, fromlist=[name])
            return getattr(module, name)
        except AttributeError:
            raise Exception("'{}' is not found in {}.\n".format(name, path))
    model = try_import(args.model, "torchvision.models")(pretrained=False)
    loss = try_import(args.loss, "torch.nn.modules.loss")()
    optimizer = try_import(args.optimizer, "torch.optim")(model.parameters(), lr=0.001)
    script_time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    t = time.time_ns()
    layer_costs, autograd_profiler = profile(model=model, loss_func=loss, optimizer=optimizer,
                                             batch_size=args.batch_size, num_of_batches=args.num_of_batches,
                                             verbosity=args.verbosity, device=args.device,
                                             enable_autograd_profiler=args.save_trace)
    t = time.time_ns()-t
    # Save reports and traces
    if args.out is None:
        file_prefix = "{}_{}".format(args.model, script_time_stamp)
    else:
        file_prefix = args.out
    out = open("{}.profile.json".format(file_prefix), "w")
    if args.verbosity >= 1:
        print("Finished in {} ms".format(t / 1e6))
        forward = 0
        backward = 0
        for layer_name, layer_dict in layer_costs.items():
            forward += sum(layer_dict["forward_pass_units"])
            backward += sum(layer_dict["backward_pass_units"])
        print("total costs in profile: {} ms".format((forward+backward)/1e6))
        print("total forward costs in profile:  {} ms ({:.2f}%)".format(forward / 1e6, forward/(forward+backward)*100))
        print("total backward costs in profile: {} ms ({:.2f}%)".format(backward / 1e6, backward/(forward+backward)*100))
    report = {"host": socket.gethostname(), "unit": "ns", "profiling_time": t, "args": args.__dict__,
              "layer_costs": layer_costs}
    json.dump(report, out, indent=4)
    out.close()
    if args.save_trace:
        autograd_profiler.export_chrome_trace("{}.chrometrace.json".format(file_prefix))
