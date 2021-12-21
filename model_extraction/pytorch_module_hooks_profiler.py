import argparse
from datetime import datetime
import json
import socket
from contextlib import nullcontext
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from pytorch_utils import *
import threading

lock = threading.Lock()

def profile(model, loss_func, optimizer, batch_size, num_of_batches, device=None, enable_autograd_profiler=False,
            verbosity=1, skip_untrainable_layers=True, check_exec_order=False, reduce_costs=False, data_path=None):
    # Setup and inject timings hooks
    if device is None:
        if torch.cuda.is_available():
            device = "gpu"
        else:
            device = "cpu"
    if device == "gpu":
        if not torch.cuda.is_available():
            raise Exception("No GPUs were detected. Change device to 'cpu' or make sure you have appropriate gpu "
                            "version of pytorch as well as the CUDA toolkit.")
        model = model.cuda()
        loss_func = loss_func.cuda()

    def init(module, input):
        with lock:
            layer_time_stamps.append((module, "init", time.time_ns()))

    def get_hook(type):
        def hook(module, input, output):
            # Forward pass has a different thread than backward pass
            # while I did not encounter any problems without the lock since the threads
            # should not be operating at the same time, I still included the lock to be safe.
            with lock:
                if device == "gpu":
                    torch.cuda.synchronize()
                layer_time_stamps.append((module, type, time.time_ns()))
        return hook

    layer_time_stamps = list()
    def register_hooks(module):
        if skip_untrainable_layers and count_trainable_params(module) == 0:
            return 0
        module.register_forward_hook(get_hook("forward"))
        module.register_backward_hook(get_hook("backward"))
        return 1

    model.register_forward_pre_hook(init)
    num_modules_processed = traverse_module(model, register_hooks)
    context = torch.autograd.profiler.profile(use_cuda=device == "gpu") if enable_autograd_profiler else nullcontext()
    # Run
    with context:
        iteration_costs = train(model=model, loss_func=loss_func, optimizer=optimizer, batch_size=batch_size,
                          num_of_batches=num_of_batches+1, device=device, verbosity=verbosity, data_path=data_path)

    # Parse costs from time stamps
    layer_costs = dict()
    module_names = list()
    for i, (module, type, ts) in enumerate(layer_time_stamps):
        if type == "init":
            continue
        module_name = get_module_name(model, module)
        module_names.append(module_name)
        if module_name not in layer_costs:
            layer_costs[module_name] = dict(forward_pass_units=list(), backward_pass_units=list(), weights_bytes=count_trainable_params(module) * 4)
        if type == "forward":
            layer_costs[module_name]["forward_pass_units"].append(ts-layer_time_stamps[i-1][2])
        elif type == "backward":
            layer_costs[module_name]["backward_pass_units"].append(ts - layer_time_stamps[i - 1][2])
    
    for module_name, costs in layer_costs.items():
        costs["forward_pass_units"] = float(np.mean(costs["forward_pass_units"][1:])) if reduce_costs  else costs["forward_pass_units"][1:]
        costs["backward_pass_units"] = float(np.mean(costs["backward_pass_units"][1:])) if reduce_costs  else costs["backward_pass_units"][1:]
    
    # Report if the execution order was constant across batches or not.
    if check_exec_order:
        # FIXME: These assertions enforce that each module is only called once in an iteration.
        # This does not necessarilly have to be true. We can edit the above parsing to allow this.
        assert len(module_names) % num_modules_processed == 0
        assert len(module_names) // 2 // num_modules_processed == num_of_batches + 1

        execution_order = module_names[:num_modules_processed]
        execution_order2 = None
        diff = False
        for i in range(1, len(module_names) // 2 // num_modules_processed):
            execution_order2 = module_names[i*num_modules_processed*2: i*num_modules_processed*2+num_modules_processed]
            if "".join(execution_order2) != "".join(execution_order):
                diff = True
                break
        if diff:
            print("Different Execution order found.\nOrder0: {}\nOrder{}: {}".format(",".join(execution_order), i, ",".join(execution_order2)))
        else:
            print("Constant Execution order found: {}".format(",".join(execution_order)))
    return layer_costs, iteration_costs, context

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
    parser.add_argument("--skip", default=False, action="store_true",
                        help="Whether or not to skip layers with no trainable parameters")
    parser.add_argument("--out", help="File name to write the final report to")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="0: Suppress all output\n"
                             "1: Show profiler output\n")
    parser.add_argument("-st", "--save_trace", default=False, action="store_true",
                        help="Whether we save the trace of the training or not. For debugging.")
    parser.add_argument("-ce", "--check_exec", default=False, action="store_true",
                        help="Whether we check that the execution order of the different modules is constant across batches")
    parser.add_argument("-rd", "--reduce_costs", default=False, action="store_true",
                        help="Whether we reduce the costs collected across different batches to a single average per layer.")
    parser.add_argument("-dt", "--data_path", default=None,
                        help="Set to None to generate synthetic data, otherwise choose the path to imagenet dataset (No other dataset is supported for now).")
    args = parser.parse_args()

    def try_import(name, path, raise_exception=True):
        try:
            module = __import__(path, fromlist=[name])
            return getattr(module, name)
        except AttributeError:
            if raise_exception:
                raise Exception("'{}' is not found in {}.\n".format(name, path))
            return False

    model = get_model(args.model)
    loss = try_import(args.loss, "torch.nn.modules.loss")()
    optimizer = try_import(args.optimizer, "torch.optim")(model.parameters(), lr=0.001)
    script_time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    t = time.time_ns()
    layer_costs, iteration_costs, autograd_profiler = profile(model=model, loss_func=loss, optimizer=optimizer,
                                                                batch_size=args.batch_size, num_of_batches=args.num_of_batches,
                                                                verbosity=args.verbosity, device=args.device,
                                                                enable_autograd_profiler=args.save_trace,
                                                                skip_untrainable_layers=args.skip,
                                                                check_exec_order=args.check_exec,
                                                                reduce_costs=args.reduce_costs,
                                                                data_path=args.data_path)
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
            forward += layer_dict["forward_pass_units"] if args.reduce_costs else sum(layer_dict["forward_pass_units"])
            backward += layer_dict["backward_pass_units"] if args.reduce_costs else sum(layer_dict["backward_pass_units"])
        print("total layer costs in profile: {} ms".format((forward+backward)/1e6))
        print("total layer forward costs in profile:  {} ms ({:.2f}%)".format(forward / 1e6, forward/(forward+backward)*100))
        print("total layer backward costs in profile: {} ms ({:.2f}%)".format(backward / 1e6, backward/(forward+backward)*100))
    report = {"method": "pytorch_module_hooks", "host": socket.gethostname(),
              "report_date": script_time_stamp, "profiling_time": t, "unit": "ns",
              "args": args.__dict__, "iteration_costs": iteration_costs, "layer_costs": layer_costs}
    json.dump(report, out, indent=4)
    out.close()
    if args.save_trace:
        autograd_profiler.export_chrome_trace("{}.chrometrace.json".format(file_prefix))
