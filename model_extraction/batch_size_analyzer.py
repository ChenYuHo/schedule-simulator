"""
A script for understanding the relationship between batch size and the total training time.
"""
import math
import sys
import os
import argparse
import time
import json
import socket
from datetime import datetime
import multiprocessing as mp


def get_pytorch_times(model, batch_size, num_of_batches, device):
    # TODO implement this
    raise Exception("Not yet implemented")


def get_tensorflow_times(model_name, batch_size, num_of_batches, device="gpu",
                         loss="binary_crossentropy", optimizer="sgd", process_queue=None, verbosity=1):
    if verbosity < 2:
        sys.stderr = open(os.devnull, 'w')
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "100"
    from tensorflow_utils import get_model, get_dummy_input_output
    import tensorflow as tf
    import tensorflow.python.keras as k
    import tensorflow.python.client.timeline as timeline
    import json

    model = get_model(model_name)
    if device == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device == "gpu":
        if not tf.test.is_gpu_available():
            raise Exception("No GPUs are available!. Change --device parameter to use cpu.")
    timings = list()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    def batch_end_callback_function(batch, log):
        trace = json.loads(timeline.Timeline(step_stats=run_metadata.step_stats).generate_chrome_trace_format())
        b, e = float("inf"), 0
        for event in trace["traceEvents"]:
            if event["ph"] == "X":
                b = min(b, event["ts"])
                e = max(e, event["ts"] + event["dur"])
        timings.append((e - b) * 1e3)  # Chrome traces are in us so we convert to ns for consistency
    callback = k.callbacks.LambdaCallback(
        on_train_batch_end=batch_end_callback_function,
    )
    x, y = get_dummy_input_output(model, batch_size)
    model.compile(optimizer=optimizer, loss=loss, options=run_options, run_metadata=run_metadata)
    model.fit(x, y, steps_per_epoch=num_of_batches, verbose=verbosity-2, callbacks=[callback])
    if process_queue is not None:
        process_queue.put(timings)
    return timings


def batch_size_analysis(model, library, device, mid_points=3, num_of_batches=10, reduce_func=None,
                        loss="binary_crossentropy", optimizer="sgd", verbosity=1):
    if library == "tensorflow":
        get_time = get_tensorflow_times
    elif library == "pytorch":
        get_time = get_pytorch_times
    else:
        raise Exception("Library '{}' is not recognized. Please choose 'tensorflow' or 'pytorch'".format(library))

    def log_msg(msg):
        if verbosity >= 1:
            print(msg)
    timings = dict()
    process_queue = mp.Queue()
    low_bs = 0
    high_bs = 0
    power = 0
    t = 0

    def get_time_process_wrapper(bs):
        nonlocal t
        log_msg("Using batch size: {}".format(bs))
        process = mp.Process(target=get_time, kwargs=dict(model_name=model, batch_size=bs,
                                                          num_of_batches=num_of_batches, device=device, loss=loss,
                                                          optimizer=optimizer, process_queue=process_queue,
                                                          verbosity=verbosity))
        process.start()
        process.join()
        if process_queue.empty():
            log_msg("Resource exhausted")
            t = None
        else:
            t = process_queue.get()
            print(t)
    while t is not None:
        for bs in range(2**power, 2**(power+1), int(math.ceil((2**(power+1)-2**power) / (mid_points + 1)))):
            get_time_process_wrapper(bs)
            if t is None:
                high_bs = bs-1
                break
            else:
                low_bs = bs+1
                timings[bs] = t
        power += 1
    while low_bs <= high_bs:
        mid_bs = (high_bs + low_bs) // 2
        get_time_process_wrapper(mid_bs)
        if t is None:
            high_bs = mid_bs - 1
        else:
            timings[mid_bs] = t
            low_bs = mid_bs + 1
    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that measures runtime vs batch size.")
    parser.add_argument("model",
                        help="The model name to analyze. Must match the name in the selected library")
    parser.add_argument("library", choices=["tensorflow", "pytorch"],
                        help="The library to use.")
    parser.add_argument("-mp", "--mid_points", type=int, default=3,
                        help="The number of mid points to have between powers of 2 batch sizes.")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                        help="The device to use for all operations. If none is specified then it is automated.")
    parser.add_argument("-nb", "--num_of_batches", type=int, default=8,
                        help="The number of batches to run")
    parser.add_argument("--out", help="File name to write the final report to")
    parser.add_argument("-l", "--loss", default="binary_crossentropy",
                        help="The loss function to use. Must match the name in the selected library.")
    parser.add_argument("-o", "--optimizer", default="sgd",
                        help="The optimizer to use when training. Must match the name in the selected library.")
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="0: Suppress all output\n"
                             "1: Show profiler output\n"
                             "2: Show tensorflow log messages\n"
                             "3: Show tensorflow function progress")
    args = parser.parse_args()
    t = time.time_ns()
    script_time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    timings = batch_size_analysis(args.model, args.library, device=args.device, mid_points=args.mid_points,
                                  num_of_batches=args.num_of_batches, verbosity=args.verbosity)
    t = time.time_ns() - t
    print("Finished in {} ms".format(t/1e6))
    if len(timings) == 0 and args.verbosity < 2:
        print("No costs were collected. Increase verbosity to find out whats happening.")
    report = {"host": socket.gethostname(),
              "report_date": script_time_stamp, "analysis_time": t, "unit": "ns",
              "args": args.__dict__, "timings": timings}
    if args.out is None:
        file_prefix = "{}_{}".format(args.model, script_time_stamp)
    else:
        file_prefix = args.out
    out = open("{}.batch_size_analysis_reports.json".format(file_prefix), "w")
    json.dump(report, out, indent=4)