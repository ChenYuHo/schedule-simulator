from schedule_simulator_core.DAGs import Layer, DAG
import math
import numpy as np

def create_DAG_from_ahmed_output(path_to_output_file, total_batch_computation_time, printer):
    printer("Creating from file '{}' with total_batch_computation_time: {}\n".format(
        path_to_output_file,total_batch_computation_time))
    root = None
    prev_layer = None
    total_tensor_sizes = 0
    layers = dict()
    with open(path_to_output_file, "r") as input_file:
        i = -1
        c = 0
        for line in input_file:
            i += 1
            try:
                sline = line.split()
                if len(sline) == 0:
                    continue
                _, _, _, full_name, _, size, size_unit, _, start_time, _, end_time, _, duration, duration_unit \
                    = sline
                size = int(size)
                duration = int(duration)
            except Exception:
                print("Problem with line {}:{}".format(i, line))
                continue
            if full_name in layers.keys():
                layers[full_name].communication_units += duration
                layers[full_name].extras['n_of_log_occurrences'] += 1
                continue
            total_tensor_sizes += size
            current_layer = Layer(tensor_size=size, is_trainable=True, communication_units=duration,
                                  name=full_name, index=c)
            layers[full_name] = current_layer
            current_layer.extras['n_of_log_occurrences'] = 1
            if root is None:
                root = current_layer
            if prev_layer is not None:
                current_layer.input_layers = [prev_layer]
                prev_layer.output_layers = [current_layer]
            prev_layer = current_layer
            c += 1
    dag = DAG(dag_input_layers=[root])
    communication_units = list()
    computation_units = list()
    for layer in layers.values():
        layer.forward_pass_units = int(math.ceil(layer.tensor_size / total_tensor_sizes * total_batch_computation_time / 2))
        layer.backward_pass_units = layer.forward_pass_units
        layer.communication_units = int(layer.communication_units / layer.extras["n_of_log_occurrences"])
        communication_units.append(layer.communication_units)
        computation_units.append(layer.forward_pass_units*2)
    printer("Number of layers: {} Number of batches: {} Total tensor size: {}\n".format(
        len(layers), layers[full_name].extras["n_of_log_occurrences"], total_tensor_sizes))
    tnet = np.array(communication_units)
    tgpu = np.array(computation_units)
    printer("Batch comp units] sum:{:5} mean:{:5.2f} std: {:5.2f}\n".format(np.sum(tgpu),np.mean(tgpu),np.std(tgpu)))
    printer("Batch comm units] sum:{:5} mean:{:5.2f} std: {:5.2f}\n".format(np.sum(tnet), np.mean(tnet), np.std(tnet)))
    return dag


if __name__ == "__main__":
    import simpy
    from schedule_simulator_core.DAGs import HomogeneousLinearDAG
    from schedule_simulator_core.core import ProcessingUnit
    from schedule_simulator_core.schedulers import FIFOScheduler, TopologicalPriorityScheduler
    from schedule_simulator_core.utils import SimPrinter, generate_ascii_timeline
    from schedule_simulator_core.DNN_functions import train
    import os

    schedulers = [FIFOScheduler(),TopologicalPriorityScheduler(preemptive=False),
                  TopologicalPriorityScheduler(preemptive=True)]
    with open("{}.simout".format(os.path.basename(__file__).split(".")[0]), "w") as file:
        dag = create_DAG_from_ahmed_output(
            "tensorflowandhorovodtraces/horovod-resnet20-cifar10-100G-node0-trace.txt", 13.6 * 1000, file.write)
        for scheduler in schedulers:
            env = simpy.Environment()
            sim_printer = SimPrinter(verbosity=0).print

            gpu = ProcessingUnit(env=env, scheduler=FIFOScheduler(), rate=1, name="GPU", sim_printer=None)
            gpu_process = env.process(gpu.main_process())

            network = ProcessingUnit(env=env, scheduler=scheduler, rate=1, name="Network", sim_printer=None)
            network_process = env.process(network.main_process())

            training_process = env.process(train(dag=dag, env=env, n_of_batches=2, batch_size=1,
                                                 computation_queue=gpu, communication_queue=network))

            def close():
                yield training_process
                print("Finishing simulation..")
                gpu_process.interrupt()
                network_process.interrupt()
            closing_process = env.process(close())
            print("Starting simulation..")
            env.run()
            units = [gpu, network]
            for unit in units:
                report = generate_ascii_timeline(unit, time_grouping=1, row_labels=["type"], cell_labels=["index"],
                                                 group_name_width=30, cell_width=5)
                file.write(report)
                file.write("\n\n\n")