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
from model_extraction.tensorflow_utils import *

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
                    warmup_batch=True):
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
        for k3 in ["count", "units", "sequential_units"]:
            stats["{}_identified_{}_percentage".format(k2, k3)] = stats["identified_{}_{}".format(k2, k3)] /\
                                                                  stats["total_{}_{}".format(k2, k3)]
    for k2 in ["forward_pass", "backward_pass"]:
        for k3 in ["count", "units", "sequential_units"]:
            stats["{}_{}_percentage".format(k2, k3)] = stats["identified_{}_{}".format(k2, k3)] /\
                                                       stats["identified_all_{}".format(k3)]


def get_pids(trace, pid_scheme):
    task_pid_names = ["/job:localhost/replica:0/task:0/device:GPU:0 Compute",
                      "/job:localhost/replica:0/task:0/device:CPU:0 Compute"]
    stream_pid_names = ["/device:GPU:0/stream:all Compute", "/device:CPU:0/stream:all Compute"]
    matched_pids = dict()
    unmatched_pids = dict()
    for event in trace["traceEvents"]:
        if event["ph"] == "M" and event["name"] == "process_name":
            pid_name = event["args"]["name"]
            if pid_scheme == "all" or pid_scheme == "task" and pid_name in task_pid_names or \
                    pid_scheme == "stream" and pid_name in stream_pid_names or \
                    pid_scheme == "task_stream" and (pid_name in task_pid_names or pid_name in stream_pid_names):
                matched_pids[pid_name] = event["pid"]
            else:
                unmatched_pids[pid_name] = event["pid"]
    return matched_pids, unmatched_pids


def parse_traces(model, traces, optimizer, pid_scheme="task", verbosity=1, skip_untrainable_layers=False):
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
    :param skip_untrainable_layers: TODO Implement this
    :return:
    """
    if skip_untrainable_layers:
        raise Exception("Skipping untrainable layers is currently not supported for layer name mapping profiling.")
    if verbosity < 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "999"
        # Supressing deprectation messages is not working
        import tensorflow.python.util.deprecation as deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False

    def log_msg(msg):
        if verbosity >= 1:
            print(msg)
    # Initialize layer_costs dict
    layer_costs = dict()
    def add_layer_name(layer):
        d = dict()
        for k2 in ["forward_pass", "backward_pass"]:
            for k3 in ["ops", "units", "sequential_units", "intervals"]:
                key = "{}_{}".format(k2, k3)
                if k3 != "intervals":
                    d[key] = [0 for _ in range(len(traces))]
                else:
                    d[key] = [None for _ in range(len(traces))]
        layer_costs[layer.name] = d
    traverse_keras_DFS(model, processing_function=add_layer_name, order="pre-order", top_to_bottom=True)
    # Initialize stats dict
    stats = dict()
    for k1 in ["total", "identified"]:
        for k2 in ["all", "forward_pass", "backward_pass"]:
            for k3 in ["count", "units", "sequential_units"]:
                stats["{}_{}_{}".format(k1, k2, k3)] = 0
    # Add the costs from all traces
    all_pids = list()
    interval_offsets = list()
    for i, trace in enumerate(traces):
        matched_pids, unmatched_pids = get_pids(trace, pid_scheme)
        if len(matched_pids) == 0:
            raise Exception("No pids were matched using the current pid_scheme. Consider changing it.\nOnly found the "
                            "following pids: {}".format(
                "\n".join(["{}: {}".format(k, v) for k, v in unmatched_pids.items()])))
        log_msg("Using pids: {}".format(matched_pids))
        all_pids.append(matched_pids)
        # First run through events. We match each event to a layer. We compute op counts, units, and intervals. O(n)
        interval_offsets.append(float("inf"))
        interval_points = set()
        filtered_events = list()
        filtered_events_matched_layers = list()
        filtered_events_types = list()
        for event in trace["traceEvents"]:
            if event["ph"] == "X" and event["pid"] in matched_pids.values():
                filtered_events.append(event)
                if event["ts"] < interval_offsets[-1]:
                    interval_offsets[-1] = event["ts"]
                interval_points.add(event["ts"])
                interval_points.add(event["ts"] + event["dur"])

                stats["total_all_count"] += 1
                stats["total_all_units"] += event["dur"]
                t = match_event_to_type(event, optimizer=optimizer)
                filtered_events_types.append(t)
                stats["total_{}_count".format(t)] += 1
                stats["total_{}_units".format(t)] += event["dur"]
                matched_layers = match_event_to_layers(layer_costs.keys(), event)
                filtered_events_matched_layers.append(matched_layers)
                if len(matched_layers) > 0:
                    stats["identified_all_count"] += 1
                    stats["identified_all_units"] += event["dur"]
                    stats["identified_{}_count".format(t)] += 1
                    stats["identified_{}_units".format(t)] += event["dur"]
                for matched_layer in matched_layers:
                    # If multiple layers are matched then we divide the event cost equally among all layers
                    layer_costs[matched_layer]["{}_ops".format(t)][i] += 1 / len(matched_layers)
                    layer_costs[matched_layer]["{}_units".format(t)][i] += event["dur"] / len(matched_layers)
                    if layer_costs[matched_layer]["{}_intervals".format(t)][i] is None:
                        layer_costs[matched_layer]["{}_intervals".format(t)][i] = event["ts"], event["ts"] + event[
                            "dur"]
                    else:
                        layer_costs[matched_layer]["{}_intervals".format(t)][i] = (
                            min(layer_costs[matched_layer]["{}_intervals".format(t)][i][0], event["ts"]),
                            max(layer_costs[matched_layer]["{}_intervals".format(t)][i][1], event["ts"] + event["dur"])
                        )
        # Second run through events. We reuse the matched layers from before and we compute sequential units. O(n^2)
        # 1. Iterate over all sub intervals.
        # 2. Find events that cover this interval.
        # 3. Find all the layers that match the events
        # 4. Split the cost of the interval between all matched layers.
        interval_points = list(interval_points)
        interval_points.sort()
        for j in range(len(interval_points)-1):
            b, e = interval_points[j], interval_points[j+1]
            cost = e - b
            matched_layers_lists = list()
            matched_events_count = 0
            types_list = list()
            found_event = False
            for k, event in enumerate(filtered_events):
                event_b, event_e = event["ts"], event["ts"] + event["dur"]
                # If this event covers the interval we are looking at
                if event_b != event_e and event_b <= b and event_e >= e:
                    found_event = True
                    if len(filtered_events_matched_layers[k]) > 0:
                        matched_events_count += 1
                    matched_layers_lists.append(filtered_events_matched_layers[k])
                    types_list.append(filtered_events_types[k])
            identified_cost = 0
            for k, matched_layers_list in enumerate(matched_layers_lists):
                assert found_event
                t = types_list[k]
                # FIXME remove this as it does not make a lot of sense
                stats["total_{}_sequential_units".format(t)] += cost / len(matched_layers_lists)
                if len(matched_layers_list) == 0:
                    continue
                sub_cost = cost / matched_events_count
                for matched_layer in matched_layers_list:
                    layer_costs[matched_layer]["{}_sequential_units".format(t)][i] += (
                            sub_cost / len(matched_layers_list)
                    )
                if len(matched_layers_list) > 0:
                    identified_cost += sub_cost
                    stats["identified_{}_sequential_units".format(t)] += sub_cost
            if found_event:
                stats["total_all_sequential_units"] += cost
                stats["identified_all_sequential_units"] += identified_cost

    # Scale layer costs & stats from microseconds to nanoseconds (For consistency)
    s = 1e3
    for key, value in stats.items():
        if "units" in key:
            stats[key] = value * s
    for layer_name, layer_dict in layer_costs.items():
        for cost_name, cost_list in layer_dict.items():
            for i, cost in enumerate(cost_list):
                if "intervals" in cost_name:
                    if cost is not None:
                        cost_list[i] = (cost[0]-interval_offsets[i])*s, (cost[1]-interval_offsets[i])*s
                elif "units" in cost_name:
                    cost_list[i] = cost * s
    # Compute percentages
    parse_stats(stats)
    return layer_costs, stats, all_pids


def profile(input_model, loss, optimizer, batch_size=32, num_of_batches=8, trials=1, verbosity=1, device="gpu",
            pid_scheme=None, skip_untrainable_layers=False):
    """
    The function takes a keras model and attempts to estimate the cost of each layer in the model in terms of
    the forward pass and backward pass.
    """
    trace_prefix = None
    traces = generate_traces(input_model, loss, optimizer, batch_size, num_of_batches, trials, verbosity, device)
    layer_costs, stats, pids = parse_traces(input_model, traces, optimizer, pid_scheme, verbosity,
                                            skip_untrainable_layers)
    if verbosity >= 1:
        for k, v in stats.items():
            if "percentage" in k:
                print("{}: {:.2f}%".format(k, v*100))
            if "total_all" in k and "units" in k:
                print("{}: {:.2f} ms".format(k, v/1e6))
    return traces, layer_costs, stats, pids


def generate_chrome_trace_for_layer_intervals(layer_costs, trace_index):
    events = list()
    for i, (layer_name, layer_dict) in enumerate(layer_costs.items()):
        fp_interval = layer_dict["forward_pass_intervals"][trace_index]
        if fp_interval is not None:
            fp_event = dict(ts=fp_interval[0]/1e3, dur=(fp_interval[1] - fp_interval[0])/1e3, name="fp_{}".format(layer_name),
                            ph="X", pid=0, tid=i*2)
            events.append(fp_event)
        bp_interval = layer_dict["backward_pass_intervals"][trace_index]
        if bp_interval is not None:
            bp_event = dict(ts=bp_interval[0]/1e3, dur=(bp_interval[1] - bp_interval[0])/1e3, name="bp_{}".format(layer_name),
                            ph="X", pid=0, tid=i*2+1)
            events.append(bp_event)
    trace = dict(traceEvents=events)
    return trace


def generate_layerwise_chrome_trace(model, trace, pid_scheme, optimizer):
    layer_names = list()

    def add_layer_name(layer):
        layer_names.append(layer.name)
    traverse_keras_DFS(model, processing_function=add_layer_name, order="pre-order", top_to_bottom=True)
    matched_pids, unmatched_pids = get_pids(trace, pid_scheme)
    if len(matched_pids) == 0:
        raise Exception("No pids were matched using the current pid_scheme. Consider changing it.\nOnly found the "
                        "following pids: {}".format(
            "\n".join(["{}: {}".format(k, v) for k, v in unmatched_pids.items()])))
    new_events = list()
    layer_indices_mapping = dict()
    # Color names available at:
    # https://github.com/catapult-project/catapult/blob/11513e359cd60e369bbbd1f4f2ef648c1bccabd0/tracing/tracing/base/color_scheme.html
    type_color_mapping = {"forward_pass": "good", "backward_pass": "terrible"}
    i = 1
    for event in trace["traceEvents"]:
        if event["ph"] == "X" and event["pid"] in matched_pids.values():
            matched_layers = match_event_to_layers(layer_names, event)
            t = match_event_to_type(event, optimizer=optimizer)
            for matched_layer in matched_layers:
                if matched_layer not in layer_indices_mapping.keys():
                    layer_indices_mapping[matched_layer] = i
                    i += 1
                new_events.append(dict(pid=0, tid=layer_indices_mapping[matched_layer], ph="X", name=event["name"],
                                       cname=type_color_mapping[t], ts=event["ts"], dur=event["dur"],
                                       args=event["args"]))
            if len(matched_layers) == 0:
                new_events.append(dict(pid=0, tid=0, ph="X", cname=type_color_mapping[t], name=event["name"],
                                       ts=event["ts"], dur=event["dur"], args=event["args"]))
    # Sort threads in topological order and name layers
    for i, layer_name in enumerate(layer_names):
        if layer_name not in layer_indices_mapping:
            continue
        new_events.append(dict(ph="M", name="thread_name", pid=0, tid=layer_indices_mapping[layer_name],
                               args=dict(name="{}_{}".format(i, layer_name))))
        new_events.append(dict(ph="M", name="thread_sort_index", pid=0, tid=layer_indices_mapping[layer_name],
                               args=dict(sort_index=i)))
    new_events.append(dict(ph="M", name="thread_name", pid=0, tid=0, args=dict(name="Unmatched ops")))
    new_events.append(dict(ph="M", name="thread_sort_index", pid=0, tid=0,args=dict(sort_index=i+1)))
    new_trace = dict(traceEvents=new_events)
    return new_trace


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
                        help="The number of training function calls to do.")
    parser.add_argument("-pi", "--pid_scheme", default="task", choices=["all", "task", "stream", "task_stream"],
                        help="The scheme in which we choose which pid groups from the trace to include in the costs."
                             "See documentation for more details.")
    parser.add_argument("--skip", default=False, action="store_true",
                        help="Whether or not to skip layers with no trainable parameters")
    parser.add_argument("--out", help="File name to write the final report to")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="0: Suppress all output\n"
                             "1: Show profiler output\n"
                             "2: Show tensorflow log messages\n"
                             "3: Show tensorflow function progress")
    parser.add_argument("-st", "--save_traces", default=False, action="store_true",
                        help="Whether we save the trace of each batch or not. For debugging.")
    parser.add_argument("-git", "--gen_interval_trace", default=False, action="store_true",
                        help="Whether or not we generate an interval trace from the last trace. For debugging.")
    parser.add_argument("-glt", "--gen_layerwise_trace", default=False, action="store_true",
                        help="Whether or not we generate a layerwise trace from the last trace. For debugging.")
    args = parser.parse_args()
    model = get_model(args.model)
    script_time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    t = time.time_ns()
    traces, layer_costs, stats, pids = profile(input_model=model, loss=args.loss, optimizer=args.optimizer,
                                               batch_size=args.batch_size, num_of_batches=args.num_of_batches,
                                               trials=args.trials, verbosity=args.verbosity, device=args.device,
                                               pid_scheme=args.pid_scheme, skip_untrainable_layers=args.skip)
    t = time.time_ns() - t
    if args.verbosity >= 1:
        print("Finished in {} ms".format(t / 1e6))
    # Save reports and traces
    if args.out is None:
        file_prefix = "{}_{}".format(args.model, script_time_stamp)
    else:
        file_prefix = args.out
    out = open("{}.profile.json".format(file_prefix), "w")
    report = {"method": "tensorflow_layer_name_mapping", "host": socket.gethostname(),
              "report_date": script_time_stamp, "profiling_time": t,  "unit": "ns",
              "args": args.__dict__, "stats": stats,
              "pids": pids, "layer_costs": layer_costs}
    json.dump(report, out, indent=4)
    out.close()
    if args.save_traces:
        for i, trace in enumerate(traces):
            with open("{}_{}.chrometrace.json".format(file_prefix, i), "w") as file:
                json.dump(trace, file, indent=4)
    if args.gen_interval_trace:
        with open("{}_intervals.chrometrace.json".format(file_prefix), "w") as f:
            json.dump(generate_chrome_trace_for_layer_intervals(layer_costs, len(traces)-1), f)
    if args.gen_layerwise_trace:
        with open("{}_layerwise.chrometrace.json".format(file_prefix), "w") as f:
            json.dump(generate_layerwise_chrome_trace(model, traces[-1], args.pid_scheme, args.optimizer), f)