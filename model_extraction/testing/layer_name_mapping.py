import json
import os
import tensorflow as tf
from model_extraction.keras_utils import traverse_keras_DFS
from model_extraction.keras_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "999"
# Supressing deprectation messages is not working
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

optimizer = "SGD"
print_op_info = False
print_pass_info = False
print_layer_names = False
new_trace = True
models = ["VGG16", "VGG19", "ResNet50", "InceptionV3", "DenseNet201"]

names_to_id = dict()
ids_to_name = dict()

def gen_trace(model):
    import tensorflow as tf
    import tensorflow.python.keras as k
    import tensorflow.python.client.timeline as timeline
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    trace = None

    def batch_end_callback_function(batch, log):
        print("batch_end_callback_function")
        nonlocal trace
        current_trace = json.loads(timeline.Timeline(step_stats=run_metadata.step_stats).generate_chrome_trace_format())
        if trace is None:
            trace = current_trace
        else:
            # extend_trace(trace, current_trace)
            trace = current_trace

    def batch_begin_callback_function(batch, log):
        pass

    callback = k.callbacks.LambdaCallback(
        on_train_batch_begin=batch_begin_callback_function, on_train_batch_end=batch_end_callback_function,
        on_predict_batch_begin=batch_begin_callback_function, on_predict_batch_end=batch_end_callback_function,
        on_test_batch_begin=batch_begin_callback_function, on_test_batch_end=batch_end_callback_function
    )
    i = 0
    def give_id(original_name):
        nonlocal i
        idd = "A18xTi{}Zhl7A4".format(i)
        names_to_id[original_name] = idd
        ids_to_name[idd] = original_name
        i += 1
        return idd
    model_dic = json.loads(model.to_json())
    for layer in model_dic["config"]["layers"]:
        idd = give_id(layer["name"])
        layer["name"] = idd
        layer["config"]["name"] = idd
    for layer in model_dic["config"]["layers"]:
        for inbound_node in layer["inbound_nodes"]:
            for la in inbound_node:
                la[0] = names_to_id[la[0]]
    for layer in model_dic["config"]["input_layers"]:
        layer[0] = names_to_id[layer[0]]
    for layer in model_dic["config"]["output_layers"]:
        layer[0] = names_to_id[layer[0]]
    model = k.models.model_from_json(json.dumps(model_dic))
    model.compile(optimizer=optimizer, loss="mean_squared_error", options=run_options, run_metadata=run_metadata)
    x, y = get_dummy_input_output(model, 2, use_numpy=True)
    model.fit(x, y, batch_size=1, callbacks=[callback], verbose=0)
    with open("{}.chrometrace.json".format(model.name), "w") as file:
        json.dump(trace, file, indent=4)
    return model, trace

for model in models:
    tf.reset_default_graph()
    with tf.Session():
        module = __import__("tensorflow.keras.applications", fromlist=[model])
        model = getattr(module, model)
        model = model(weights=None, include_top=True)
        print("Model: {}".format(model.name))

        if new_trace:
            model, trace = gen_trace(model)
        else:
            with open("{}.chrometrace.json".format(model.name)) as file:
                trace = json.load(file)


    layer_costs = dict()

    op_counts = dict(total=dict(), identified=dict(), forward_pass=dict(), identified_forward_pass=dict(),
                     backward_pass=dict(), identified_backward_pass=dict())
    op_durations = dict(total=dict(), identified=dict(), forward_pass=dict(), identified_forward_pass=dict(),
                        backward_pass=dict(), identified_backward_pass=dict())
    def add_layer_name(layer):
        layer_costs[layer.name] = dict(communication_units=layer.count_params() * 4, forward_pass_units=0,
                                       backward_pass_units=0, forward_pass_ops=0, backward_pass_ops=0)
    def add(dic, key, value):
        if key in dic.keys():
            dic[key] += value
        else:
            dic[key] = value

    def filter_matched_layers(se):
        li = list(se)
        li.sort(key=len)
        filtered_list = set()
        for i, name in enumerate(li):
            subset = False
            for j in range(i+1, len(li)):
                if name in li[j]:
                    subset = True
            if not subset:
                filtered_list.add(name)
        return filtered_list
    def match_to_layer(event):
        matched_layers = set()
        for layer_name in layer_costs.keys():
            if layer_name in event["args"]["name"]:
                matched_layers.add(layer_name)
        if len(matched_layers) == 0:
            # Let us look into the event inputs
            for key, value in event["args"].items():
                if "input" not in key:
                    continue
                ml = set()
                for layer_name in layer_costs.keys():
                    if layer_name in value:
                        ml.add(layer_name)
                matched_layers.update(ml)
        else:
            matched_layers = matched_layers
        if len(matched_layers) > 1:
            print(event)
            print(matched_layers)
        return matched_layers

    def op(event):
        return event["args"]["op"].split("#")[0]

    traverse_keras_DFS(model, processing_function=add_layer_name, order="pre-order", top_to_bottom=True)
    if print_layer_names:
        print(list(layer_costs.keys()))
    multi_matched_events = 0
    for event in trace["traceEvents"]:
        if event["ph"] == "X" and event["pid"]:
            add(op_counts["total"], op(event), 1)
            add(op_durations["total"], op(event), event["dur"])
            t = "backward_pass" if optimizer in event["args"]["name"] else "forward_pass"
            add(op_counts[t], op(event), 1)
            add(op_durations[t], op(event), event["dur"])
            matched_layers = match_to_layer(event)
            if len(matched_layers) > 0:
                add(op_counts["identified"], op(event), 1)
                add(op_durations["identified"], op(event), event["dur"])
                add(op_counts["identified_{}".format(t)], op(event), 1)
                add(op_durations["identified_{}".format(t)], op(event), event["dur"])
            for matched_layer in matched_layers:
                layer_costs[matched_layer]["{}_ops".format(t)] += 1
                layer_costs[matched_layer]["{}_units".format(t)] += event["dur"] / len(matched_layers)
            # if len(matched_layers) > 1:
            #     print("Layers: {} Event: {}".format(matched_layers, event["args"]))
    if multi_matched_events > 0:
        print("Warning {}/{} events have been matched to multiple layers.".format(multi_matched_events,
                                                                                  sum(op_counts["total"].values())))

    # Print stats
    for name, dic in [("ops", op_counts), ("durations", op_durations)]:
        print("Identified {}: {:.2f}%".format(name, sum(dic["identified"].values()) / sum(dic["total"].values()) * 100))
        if not print_pass_info:
            continue
        print("Identified forward pass {}: {:.2f}%".format(name, sum(dic["identified_forward_pass"].values()) /
                                                          sum(dic["forward_pass"].values()) * 100))
        print("Identified backward pass {}: {:.2f}%".format(name, sum(dic["identified_backward_pass"].values()) /
                                                           sum(dic["backward_pass"].values()) * 100))
        print("Forward pass {}: {:.2f}%".format(name, sum(dic["forward_pass"].values()) / sum(dic["total"].values()) * 100))
        print("Backward pass {}: {:.2f}%".format(name, sum(dic["backward_pass"].values()) / sum(dic["total"].values()) * 100))
        print()

    if print_op_info:
        for t in ["forward_pass", "backward_pass"]:
            ident_keys = set(op_counts["identified_{}".format(t)].keys())
            all_keys = set(op_counts[t].keys())
            print("Identified {} op types: {}".format(t, ident_keys))
            print("Unidentified {} op types: {}".format(t, all_keys - ident_keys))
        print()
        for op in op_counts["total"].keys():
            print("Op: {}---------------".format(op))
            print("Identified ops: {:.2f}% Identified durations: {:.2f}%".format(
                op_counts["identified"][op] / op_counts["total"][op] * 100 if op in op_counts["identified"] else 0,
                op_durations["identified"][op] / op_durations["total"][op] * 100 if op in op_durations["identified"] else 0
            ))
            print("Composes {:.2f}% of total ops. Composes {:.2f}% of total durations.".format(
                op_counts["total"][op] / sum(op_counts["total"].values()) * 100 ,
                op_durations["total"][op] / sum(op_durations["total"].values()) * 100
            ))
            print("Composes {:.2f}% of identified ops. Composes {:.2f}% of identified durations.".format(
                op_counts["identified"][op] / sum(op_counts["identified"].values()) * 100 if op in op_counts["identified"] else 0,
                op_durations["identified"][op] / sum(op_durations["identified"].values()) * 100  if op in op_counts["identified"] else 0,
            ))
