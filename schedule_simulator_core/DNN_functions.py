"""
This module simulates the functions of a DNN (Training, inference, ...etc).
Building the architecture is done in the DAGs module. Using the DAG architecture as a DNN is done here
"""

import simpy
from simpy.events import AllOf
from schedule_simulator_core.core import Job
from schedule_simulator_core.DAGs import DAG


def train(dag: DAG, env: simpy.Environment, n_of_batches, batch_size, computation_queue, communication_queue,
          **job_extras):
    """
    :param dag: The architecture to use
    :param env: The simpy environment used in this simulation.
    :param n_of_batches: How many batches should we run
    :param batch_size: How many forward & backward passes before we need to update the parameters or synchronize.
    :param computation_queue: Used to queue computational jobs
    :param communication_queue: Used to queue communication jobs
    :param job_extras: Extras to be passed to all created jobs
    """
    last_forward_pass_output = None
    last_backward_pass_output = None
    for batch in range(n_of_batches):
        extras = job_extras.copy()
        extras['batch'] = batch
        last_forward_pass_output = yield env.process(forward_pass(dag=dag, env=env, batch_size=batch_size,
                                                                  computation_queue=computation_queue,
                                                                  communication_queue=communication_queue,
                                                                  dependent_layer_jobs=last_backward_pass_output,
                                                                  **extras))
        last_backward_pass_output = yield env.process(backward_pass(dag=dag, env=env, batch_size=batch_size,
                                                                    computation_queue=computation_queue,
                                                                    communication_queue=communication_queue,
                                                                    dependent_layer_jobs=last_forward_pass_output,
                                                                    send_gradients=True, **extras))
    # Finish any impending processes
    for process in last_backward_pass_output.values():
        yield process
    for process in last_backward_pass_output.values():
        yield process


def forward_pass(dag: DAG, env: simpy.Environment, batch_size, computation_queue, communication_queue, dependent_layer_jobs=None,
                 **job_extras):
    """
    :param dag: The architecture to use
    :param env: The simpy environment used in this simulation.
    :param computation_queue: Used to queue computational jobs
    :param communication_queue: Used to queue communication jobs
    :param dependent_layer_jobs: A dict with key: layer and value: event to wait for before being able to do a
    forward pass on that layer. If none then we simply do not wait.
    :param job_extras: Extras to be passed to all created jobs
    :return: A dict with key: layer and value: event to wait for before being able to do a backward pass on that layer
    """
    forward_pass_output = dict()
    for layer in dag.topological_order:
        # Wait for all dependencies to finish
        if dependent_layer_jobs is not None:
            deps = [layer]  # Must include the layer itself
            deps.extend(layer._forward_dependencies)
            for dep in deps:
                yield dependent_layer_jobs[dep]
        # Create job
        extras = {**layer.extras, **job_extras}  # add layer extras to custom extras passed to the function
        extras["type"] = "forward_pass"
        job = Job(env, layer.forward_pass_units*batch_size, source=layer, **extras)
        forward_pass_output[layer] = job
        # Queue job
        computation_queue.queue(job)
        # FIXME
        # Waiting for the job before queueing the next makes sense because the next job depends on this one.
        # A problem arises however when the processing rate can accommodate more than a job at a time step.
        # Since the next job won't be queued until the next time step, we will not be able to utilize the processing
        # unit fully. To solve this we have a couple of options:
        # - (X) Peek into the processing unit to learn how many jobs it can execute in the next time step.
        # - Always make job units divisible by the rate
        # - Implement job chaining where dependent jobs are linked together through the job result field
        # - Add a dependencies field to the Job class and force schedulers to respect those dependencies.
        yield job
    return forward_pass_output


def backward_pass(dag: DAG, env: simpy.Environment, batch_size, computation_queue, communication_queue, dependent_layer_jobs=None,
                  send_gradients=False, **job_extras):
    """
    :param dag: The architecture to use
    :param env: The simpy environment used in this simulation.
    :param computation_queue: Used to queue computational jobs
    :param communication_queue: Used to queue communication jobs
    :param dependent_layer_jobs: A dict with key: layer and value: event to wait for before being able to do a
    backward pass on that layer. If none then we simply do not wait.
    :param send_gradients: Whether we should queue gradients as they are produced to the communication_queue. Should
    only be set to true at the last backward_pass in a batch
    :return: A dict with key: layer and value: event to wait for before being able to do a forward pass on that
    layer
    """
    backward_pass_output = dict()
    reversed_topological_order = dag.topological_order.copy()
    reversed_topological_order.reverse()
    for layer in reversed_topological_order:
        # Wait for all dependencies to finish
        if dependent_layer_jobs is not None:
            deps = [layer]  # Must include the layer itself
            deps.extend(layer._backward_dependencies)
            for dep in deps:
                yield dependent_layer_jobs[dep]
        # Create job
        comp_extras = {**layer.extras, **job_extras}  # add layer extras to custom extras passed to the function
        comm_extras = comp_extras.copy()
        comp_extras["type"] = "backward_pass"
        comp_job = Job(env, layer.backward_pass_units*batch_size, source=layer, **comp_extras)
        if send_gradients:
            comm_extras["type"] = "parameter_communication"
            comm_job = Job(env, layer.communication_units, source=layer, **comm_extras)
            backward_pass_output[layer] = AllOf(env, [comm_job, comp_job])
        else:
            backward_pass_output[layer] = comp_job
        # We only wait for the computational job.
        computation_queue.queue(comp_job)
        # Same problem of waiting as mentioned in the comments in forward pass function
        yield comp_job
        if send_gradients:
            communication_queue.queue(comm_job)
    return backward_pass_output


if __name__ == "__main__":
    """
    An example usage where different schedulers are compared
    """
    from schedule_simulator_core.DAGs import HomogeneousLinearDAG
    from schedule_simulator_core.core import ProcessingUnit
    from schedule_simulator_core.schedulers import FIFOScheduler, TopologicalPriorityScheduler
    from schedule_simulator_core.utils import SimPrinter, generate_chrome_trace_timeline, join_chrome_traces
    schedulers = [FIFOScheduler(),TopologicalPriorityScheduler(preemptive=False),
                  TopologicalPriorityScheduler(preemptive=True)]
    units = list()
    for scheduler in schedulers:
        env = simpy.Environment()
        sim_printer = SimPrinter(verbosity=0).print
        dag = HomogeneousLinearDAG(n_of_layers=6, fp_units=8, bp_units=8, comm_units=8)

        gpu = ProcessingUnit(env=env, scheduler=FIFOScheduler(), rate=4, name="GPU_{}".format(scheduler),
                             sim_printer=None, timeline_format="jobwise")
        gpu_process = env.process(gpu.main_process())

        network = ProcessingUnit(env=env, scheduler=scheduler, rate=1, name="Network_{}".format(scheduler),
                                 sim_printer=None, timeline_format="jobwise")
        network_process = env.process(network.main_process())

        training_process = env.process(train(dag=dag, env=env, n_of_batches=10, batch_size=2,
                                             computation_queue=gpu, communication_queue=network))

        def close():
            yield training_process
            print("Finishing simulation..")
            gpu_process.interrupt()
            network_process.interrupt()
        closing_process = env.process(close())
        print("Starting simulation..")
        env.run()
        units.extend([gpu, network])
    traces = list()
    for unit in units:
        traces.append(generate_chrome_trace_timeline(unit, group_labels=["unit_name"], row_labels=["type"],
                                                     cell_labels=["name"], utilization_bins=500))
    final_trace = join_chrome_traces(traces)
    with open("DNN_functions_example_trace.json", "w") as file:
        file.write(final_trace)
