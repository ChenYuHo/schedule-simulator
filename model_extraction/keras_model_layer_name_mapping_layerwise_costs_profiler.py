"""
A script for getting layer wise timings about any Keras model using the tensorflow tracer by mapping layer names to
the operations.

This script relies on the keras_utils file. Including that file in the same directory when running this script will do
the trick.
"""

import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.client.timeline as timeline
import os
import sys
import argparse
from datetime import datetime
import socket
sys.path.append("..")
from model_extraction.keras_utils import *

# A random sequence of characters to use in wrapping layer names to ensure they can be identified even
# after tensorflow adds its own prefixes and suffixes
IDENTIFYING_STRING = "hXa81Oz0"


def wrap_name(name):
    return "{}{}{}".format(IDENTIFYING_STRING, name, IDENTIFYING_STRING)


def extract_name(string):
    b = string.find(IDENTIFYING_STRING)
    e = string[b+1:].find(IDENTIFYING_STRING)
    return string[b+len(IDENTIFYING_STRING):e+b+1]


def wrap_layer_names(model):
    model_dic = json.loads(model.to_json())
    for layer in model_dic["config"]["layers"]:
        new_name = wrap_name(layer["name"])
        layer["name"] = new_name
        layer["config"]["name"] = new_name
    for layer in model_dic["config"]["layers"]:
        for inbound_node in layer["inbound_nodes"]:
            for la in inbound_node:
                la[0] = wrap_name(la[0])
    for layer in model_dic["config"]["input_layers"]:
        layer[0] = wrap_name(layer[0])
    for layer in model_dic["config"]["output_layers"]:
        layer[0] = wrap_name(layer[0])
    return k.models.model_from_json(json.dumps(model_dic))


def generate_traces(input_model, loss, optimizer, batch_size=32, num_of_batches=8, trials=1, verbosity=1, device="gpu",
                    output_file_prefix=None, warmup_batch=True):
    # Early checks and setup -------------------------------------------------------------------------------------------
    if tf.executing_eagerly():
        raise Exception("Cannot use tracer with eager execution.")
    if verbosity < 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "999"
        # Supressing deprectation messages is not working
        import tensorflow.python.util.deprecation as deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False

    def log_msg(msg):
        if verbosity >= 1:
            print(msg)
    # Check device
    if device == "cpu":
        # This seems to be the most effective method. Other methods include using the tf.device("/cpu:0")
        # context manager or setting ConfigProto(device_count={'GPU':0}. But both these methods use the GPU a little bit
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        log_msg("Running on CPU")
    elif device == "gpu":
        if not tf.test.is_gpu_available():
            raise Exception("No GPUs are available!. Change --device parameter to use cpu.")
        log_msg("Running on GPU")
    # Wrap layer names to ensure they can be identified even after tensorflow adds its own prefixes and suffixes.
    model = wrap_layer_names(input_model)
    # Run and generate traces ------------------------------------------------------------------------------------------
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    traces = list()
    def batch_end_callback_function(batch, log):
        traces.append(timeline.Timeline(step_stats=run_metadata.step_stats).generate_chrome_trace_format())
    callback = k.callbacks.LambdaCallback(
        on_train_batch_end=batch_end_callback_function,
    )
    model.compile(optimizer=optimizer, loss=loss, options=run_options, run_metadata=run_metadata)
    x, y = get_dummy_input_output(input_model, batch_size)
    if warmup_batch:
        model.fit(x, y, steps_per_epoch=1, verbose=0)
    for _ in range(trials):
        model.fit(x, y, steps_per_epoch=num_of_batches, callbacks=[callback], verbose=verbosity)
    for i, trace in enumerate(traces):
        traces[i] = json.loads(trace)
    if output_file_prefix is not None:
        for i, trace in enumerate(traces):
            with open("{}_{}.chrometrace.json".format(output_file_prefix, i), "w") as file:
                json.dump(trace, file, indent=4)
    return traces


def match_event_to_layers(layer_names, event):
    matched_layers = set()
    for layer_name in layer_names:
        if wrap_name(layer_name) in event["args"]["name"]:
            matched_layers.add(layer_name)
    if len(matched_layers) == 0:
        # Let us look into the event inputs
        # TODO I should make this a recursive call that traces the inputs until identifying the event
        for key, value in event["args"].items():
            if "input" not in key:
                continue
            ml = set()
            for layer_name in layer_names:
                if wrap_name(layer_name) in value:
                    ml.add(layer_name)
            matched_layers.update(ml)
    else:
        matched_layers = matched_layers
    return matched_layers


def match_event_to_type(event, **kwargs):
    if kwargs["optimizer"].upper() in event["args"]["name"] or kwargs["optimizer"] in event["args"]["name"]:
        return "backward_pass"
    else:
        return "forward_pass"


def parse_stats(stats):
    for k2 in ["all", "forward_pass", "backward_pass"]:
        for k3 in ["count", "units", "durations"]:
            stats["{}_identified_{}_percentage".format(k2, k3)] = stats["identified_{}_{}".format(k2, k3)] /\
                                                                  stats["total_{}_{}".format(k2, k3)]
    for k2 in ["forward_pass", "backward_pass"]:
        for k3 in ["count", "units", "durations"]:
            stats["{}_{}_percentage".format(k2, k3)] = stats["identified_{}_{}".format(k2, k3)] /\
                                                       stats["identified_all_{}".format(k3)]


def parse_traces(model, traces, optimizer, pid_scheme="task", verbosity=1):
    """

    :param model:
    :param traces:
    :param optimizer:
    :param pid_scheme: The scheme in which we choose which pid groups from the trace to include in the costs
    'task': use "/job:localhost/replica:0/task:0/device:GPU:0 Compute" &
    "/job:localhost/replica:0/task:0/device:CPU:0 Compute"
    'stream: use "/device:GPU:0/stream:all Compute" & "/device:CPU:0/stream:all Compute"
    "task_stream" use both task names and stream names
    'all: use all pids
    :param verbosity:
    :return:
    """
    task_pid_names = ["/job:localhost/replica:0/task:0/device:GPU:0 Compute",
                      "/job:localhost/replica:0/task:0/device:CPU:0 Compute"]
    stream_pid_names = ["/device:GPU:0/stream:all Compute", "/device:CPU:0/stream:all Compute"]
    if verbosity < 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "999"
        # Supressing deprectation messages is not working
        import tensorflow.python.util.deprecation as deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False

    def log_msg(msg):
        if verbosity >= 1:
            print(msg)
    layer_costs = dict()

    def add_layer_name(layer):
        d = dict()
        for key in ["forward_pass_units", "backward_pass_units", "forward_pass_ops", "backward_pass_ops",
                    "forward_pass_durations", "backward_pass_durations"]:
            d[key] = [0 for _ in range(len(traces))]
        for key in ["forward_pass_intervals", "backward_pass_intervals"]:
            d[key] = [None for _ in range(len(traces))]
        layer_costs[layer.name] = d
    traverse_keras_DFS(model, processing_function=add_layer_name, order="pre-order", top_to_bottom=True)
    stats = dict()
    def update_interval(stats, key, event):
        interval_key = key + "_intervals"
        duration_key = key + "_durations"
        if interval_key not in stats:
            stats[interval_key] = event["ts"], (event["ts"] + event["dur"])
        else:
            stats[interval_key] = (min(stats[interval_key][0], event["ts"]),
                                   max(stats[interval_key][1], event["ts"] + event["dur"]))
        stats[duration_key] = stats[interval_key][1] - stats[interval_key][0]
    for k1 in ["total", "identified"]:
        for k2 in ["all", "forward_pass", "backward_pass"]:
            for k3 in ["count", "units", "durations"]:
                stats["{}_{}_{}".format(k1, k2, k3)] = 0
    # Add the costs from all traces
    all_pids = list()
    for i, trace in enumerate(traces):
        current_pids = dict()
        for event in trace["traceEvents"]:
            if event["ph"] == "M" and event["name"] == "process_name":
                pid_name = event["args"]["name"]
                if pid_scheme == "all" or pid_scheme == "task" and pid_name in task_pid_names or \
                        pid_scheme == "stream" and pid_name in stream_pid_names or \
                        pid_scheme == "task_stream" and (pid_name in task_pid_names or pid_name in stream_pid_names):
                    current_pids[event["args"]["name"]] = event["pid"]
        if len(current_pids) == 0:
            raise Exception("No pids were matched using the current pid_scheme.Consider changing it.")
        log_msg("Using pids: {}".format(current_pids))
        all_pids.append(current_pids)
        for event in trace["traceEvents"]:
            if event["ph"] == "X" and event["pid"] in current_pids.values():
                stats["total_all_count"] += 1
                stats["total_all_units"] += event["dur"]
                update_interval(stats, "total_all", event)
                t = match_event_to_type(event, optimizer=optimizer)
                stats["total_{}_count".format(t)] += 1
                stats["total_{}_units".format(t)] += event["dur"]
                update_interval(stats, "total_{}".format(t), event)
                matched_layers = match_event_to_layers(layer_costs.keys(), event)
                if len(matched_layers) > 0:
                    stats["identified_all_count"] += 1
                    stats["identified_all_units"] += event["dur"]
                    update_interval(stats, "identified_all", event)
                    stats["identified_{}_count".format(t)] += 1
                    stats["identified_{}_units".format(t)] += event["dur"]
                    update_interval(stats, "identified_{}".format(t), event)
                for matched_layer in matched_layers:
                    # If multiple layers are matched then we divide the event cost equally among all layers
                    layer_costs[matched_layer]["{}_ops".format(t)][i] += 1 / len(matched_layers)
                    layer_costs[matched_layer]["{}_units".format(t)][i] += event["dur"] / len(matched_layers)
                    if layer_costs[matched_layer]["{}_intervals".format(t)][i] is None:
                        layer_costs[matched_layer]["{}_intervals".format(t)][i] = event["ts"], event["ts"] + event["dur"]
                    else:
                        layer_costs[matched_layer]["{}_intervals".format(t)][i] = (
                            min(layer_costs[matched_layer]["{}_intervals".format(t)][i][0], event["ts"]),
                            max(layer_costs[matched_layer]["{}_intervals".format(t)][i][1], event["ts"] + event["dur"])
                        )
                    layer_costs[matched_layer]["{}_durations".format(t)][i] = \
                        layer_costs[matched_layer]["{}_intervals".format(t)][i][1] - \
                        layer_costs[matched_layer]["{}_intervals".format(t)][i][0]

    # Scale layer costs & stats from microseconds to nanoseconds (For consistency)
    s = 1e3
    for key, value in stats.items():
        if "intervals" in key:
            stats[key] = value[0] * s, value[1] * s
        else:
            stats[key] = value * s
    for layer_name, layer_dict in layer_costs.items():
        for cost_name, cost_list in layer_dict.items():
            for i, cost in enumerate(cost_list):
                if "intervals" in cost_name:
                    if cost is not None:
                        cost_list[i] = cost[0] * s, cost[1] * s
                else:
                    cost_list[i] = cost * s
    # Compute percentages
    parse_stats(stats)
    return layer_costs, stats, all_pids


def profile(input_model, loss, optimizer, batch_size=32, num_of_batches=8, trials=1, verbosity=1, device="gpu",
            pid_scheme=None, save_traces=False):
    """
    The function takes a keras model and attempts to estimate the cost of each layer in the model in terms of
    the forward pass and backward pass.
    """
    trace_prefix = None
    if save_traces:
        trace_prefix = "{}_{}".format(args.model, SCRIPT_START.strftime("%m-%d-%H-%M"))
    traces = generate_traces(input_model, loss, optimizer, batch_size, num_of_batches, trials, verbosity, device,
                             output_file_prefix=trace_prefix)
    layer_costs, stats, pids = parse_traces(input_model, traces, optimizer, pid_scheme, verbosity)
    if verbosity >= 1:
        for k, v in stats.items():
            if "percentage" in k:
                print("{}: {:.2f}%".format(k, v*100))
    return layer_costs, stats, pids


def generate_chrome_trace_for_layer_intervals(layer_costs, trace_index):
    events = list()
    for i, (layer_name, layer_dict) in enumerate(layer_costs.items()):
        fp_interval = layer_dict["forward_pass_intervals"][trace_index]
        print(fp_interval[1] - fp_interval[0] - layer_dict["forward_pass_durations"][trace_index])
        assert fp_interval[1] - fp_interval[0] - layer_dict["forward_pass_durations"][trace_index] < 1e-9
        fp_event = dict(ts=fp_interval[0], dur=fp_interval[1] - fp_interval[0], name="fp_{}".format(layer_name),
                        ph="X", pid=0, tid=i*2)
        events.append(fp_event)
        bp_interval = layer_dict["backward_pass_intervals"][trace_index]
        assert bp_interval[1] - bp_interval[0] - layer_dict["backward_pass_durations"][trace_index] < 1e-9
        bp_event = dict(ts=bp_interval[0], dur=bp_interval[1] - bp_interval[0], name="bp_{}".format(layer_name),
                        ph="X", pid=0, tid=i*2+1)
        events.append(bp_event)
    trace = dict(traceEvents=events)
    return trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that profiles Keras models and produces layer wise timings.")
    parser.add_argument("model",
                        help="The Keras model to profile. See Keras.Applications documentation for all options."
                             "Dummy options include: [dummy_multi, dummy_linear_dense, dummy_linear_cnn, dummy_2_layers]")
    parser.add_argument("-l", "--loss", default="binary_crossentropy",
                        help="The loss function to use. See Keras.Losses documentation for all options.")
    parser.add_argument("-o", "--optimizer", default="sgd",
                        help="The optimizer to use when training. See Keras.Optimizers documentation for all options.")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="The device to use for all operations. If none is specified then it is automated.")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="The batch size used for all functions")
    parser.add_argument("-nb", "--num_of_batches", type=int, default=8,
                        help="The number of batches to run")
    parser.add_argument("-t", "--trials", type=int, default=1,
                        help="The number of layer building & evaluation iterations to do.")
    parser.add_argument("-pi", "--pid_scheme", choices=["all", "task", "stream", "task_stream"],
                        help="The scheme in which we choose which pid groups from the trace to include in the costs."
                             "See documentation for more details.")
    parser.add_argument("--out",
                        help="Stream to write the timings to in json format if any. File | stdout | stderr | suppress")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="0: Suppress all output\n"
                             "1: Show profiler output\n"
                             "2: Show tensorflow log messages\n"
                             "3: Show tensorflow function progress")
    parser.add_argument("-st", "--save_traces", default=False, action="store_true",
                        help="Whether we save the trace of each batch or not. For debugging.")
    parser.add_argument("-git", "--gen_interval_trace", default=False, action="store_true",
                        help="Whether we save a trace of the layer intervals or not")
    args = parser.parse_args()
    try:
        model_func_name = "{}_model".format(args.model)
        if model_func_name in dir(sys.modules[__name__]):
            module = sys.modules[__name__]
            model = getattr(module, model_func_name)()
        else:
            module = __import__("tensorflow.keras.applications", fromlist=[args.model])
            model = getattr(module, args.model)
            model = model(weights=None, include_top=True)
    except AttributeError:
        raise Exception("'{}' is not a valid dummy or keras model.\n".format(args.model))
    SCRIPT_START = datetime.now()
    layer_costs, stats, pids = profile(input_model=model, loss=args.loss, optimizer=args.optimizer,
                                       batch_size=args.batch_size, num_of_batches=args.num_of_batches,
                                       trials=args.trials, verbosity=args.verbosity, device=args.device,
                                       pid_scheme=args.pid_scheme, save_traces=args.save_traces)
    if args.out is not None:
        if args.out == "stdout":
            out = sys.__stdout__
        elif args.out == "stderr":
            out = sys.__stderr__
        elif args.out == "suppress":
            out = False
        else:
            out = open(args.out, "w")
    else:
        out = open("{}_{}.profile.json".format(args.model, SCRIPT_START.strftime("%m-%d-%H-%M")), "w")
    report = {"host": socket.gethostname(), "unit": "ns", "args": args.__dict__, "stats": stats, "pids": pids,
              "layer_costs": layer_costs}
    json.dump(report, out, indent=4)
    out.close()
    if args.gen_interval_trace:
        with open("{}_{}_intervals.chrometrace.json".format(args.model, SCRIPT_START.strftime("%m-%d-%H-%M")), "w") as f:
            json.dump(generate_chrome_trace_for_layer_intervals(layer_costs, 0), f)