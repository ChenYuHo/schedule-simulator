import json
import numpy as np


class SimPrinter:
    def __init__(self, verbosity=2):
        self.verbosity = verbosity

    def print(self, env, source, msg, verbosity):
        if self.verbosity >= verbosity:
            print("t:{:<6.2f}] {:<10}] {}".format(env.now, str(source), msg))


def group_dict(dictionary, key_indices, extend_existing_lists=True):
    """
    A general method that collapses or groups a flat dictionary with compound keys,into a dictionary with smaller
    compound keys and a list of values.
    Ex.
    A = { ("O",1): [10,20], ("A",1): 5, ("O",2): 30 }
    group_dict(A, key_indices=[0]) returns {(1,): [10,20,5], (2,): [30]}
    group_dict(A, key_indices=[1]) returns {('O',): [10,20,30], ('A',): [5]}
    group_dict(A, key_indices=[0,1]) {(): [10,20,5,30]}
    group_dict(A, key_indices=[]) returns {('O', 1): [10,20], ('A', 1): [5], ('O', 2): [30]}
    :param dictionary: The dictionary to group
    :param key_indices: The key indices to DROP.
    :param extend_existing_lists: If true then a any value that is an iterable will be extended.
    Ex. Joining [1,2,3] 2 would result in [1,2,3,2]
    If false, then a new list would be constructed that joins those values.
    Ex. Joining [1,2,3] 2 would result in [[1,2,3], 2]
    """
    result = dict()
    for compound_key in dictionary.keys():
        filtered_key = tuple([x for i, x in enumerate(compound_key) if i not in key_indices])
        value = dictionary[compound_key]
        if filtered_key not in result:
            if extend_existing_lists:
                try:
                    result[filtered_key] = list(value)
                except TypeError:
                    result[filtered_key] = [value]
            else:
                result[filtered_key] = [value]
        else:
            if extend_existing_lists:
                try:
                    result[filtered_key].extend(value)
                except TypeError:
                    result[filtered_key].append(value)
            else:
                result[filtered_key].append(value)
    return result


TIME_UNITS = dict(d=60*60*24, h=60*60, m=60, s=1, ms=1e-3, us=1e-6, ns=1e-9, ps=1e-12)
BYTE_UNITS = dict(kB=1e3, kiB=2**10, MB=1e6, MiB=2**20, GB=1e9, GiB=2**30, TB=1e12, TiB=2**40)
BIT_UNITS = {k[:-1]+"b": v*8 for k, v in BYTE_UNITS.items()}


def convert_time(value, from_unit, to_unit):
    return value * (TIME_UNITS[from_unit] / TIME_UNITS[to_unit])


def Mbps_to_Bpns(value):
    # a = value / 8 > MBps, a = a * 1e6 > Bps, a = a * 1e-9 > Bpns
    return value / 8 * 1e6 * 1e-9


def sort_table(table, key):
    """
    :param table: dict(key="row_name", value=list of values)
    :param key: The row_name to sort by
    """
    for row_name in table.keys():
        if row_name == key:
            continue
        table[row_name] = [x for _, x in sorted(zip(table[key], table[row_name]))]
    table[key].sort()


def join_overlapping_intervals(intervals_list):
    """
    :param intervals_list: A list of (begin, end) tuples that represent intervals
    :return: A mutually exclusive (No overlapping) reduced list of intervals.
    """
    intervals_list.sort(key=lambda x: x[0])
    reduced_intervals_stack = list()
    for interval in intervals_list:
        if len(reduced_intervals_stack) == 0:
            reduced_intervals_stack.append(interval)
            continue
        last_start, last_end = reduced_intervals_stack[-1]
        current_start, current_end = interval
        if current_start <= last_end:
            reduced_intervals_stack[-1] = (last_start, max(current_end, last_end))
        else:
            reduced_intervals_stack.append(interval)
    return reduced_intervals_stack


def get_interval_gaps(intervals_list):
    """
    :param intervals_list: A list of (begin, end) tuples that represent intervals
    :return: A mutually exclusive (No overlapping) list of gaps between the intervals.
    """
    intervals_list.sort(key=lambda x: x[0])
    gaps_list = [(intervals_list[0][0], max(intervals_list, key=lambda x: x[1])[1])]
    for interval in intervals_list:
        if len(gaps_list) == 0:
            break
        last_gap_start, last_gap_end = gaps_list[-1]
        current_interval_start, current_interval_end = interval
        if current_interval_start >= last_gap_end:
            break
        if current_interval_end > last_gap_start:
            if current_interval_start > last_gap_start:
                first_gap = (last_gap_start, current_interval_start)
                second_gap = (current_interval_end, last_gap_end)
                gaps_list[-1] = first_gap
                if second_gap[1] - second_gap[0] > 0:
                    gaps_list.append(second_gap)
            else:
                if current_interval_end == last_gap_end:
                    gaps_list.pop()
                else:
                    gaps_list[-1] = (current_interval_end, last_gap_end)
    return gaps_list


def trim(string, length):
    string = str(string)
    if len(string) > length:
        return string[:length-2] + ".."
    else:
        return string


def get_job_group(job, extras_keys, **patch_extras):
    if extras_keys is None or len(extras_keys) == 0:
        return tuple()
    extras = job.extras.copy()
    extras.update(patch_extras)
    group = list()
    for key in extras_keys:
        if key in extras:
            group.append(extras[key])
        else:
            group.append(None)
    return tuple(group)


def generate_ascii_timeline(processing_unit, start=0, end=None,
                            time_grouping=1, show_scaled_time=False,
                            row_labels=None, cell_labels=None,
                            show_column_labels=True, show_row_utilization=True, show_header=True, suppress_body=False,
                            long_value_handling="trim", cell_width=5, group_name_width=20):
    """
    The current implementation is a little messy and inefficient just to get quick insight.
    It should be refined and more structured later.
    Also note that this function does not use the get_utilization method. Instead it implements its own so that the
    utilization is calculated alongside the generation of the report.
    :param processing_unit: The unit to generate the report on. Will accept a list of units in a later implementation
    :param start: From which time step should we generate the report (Inclusive)
    :param end: To which time step (Inclusive)
    :param time_grouping: How many time steps per column. If set to 0 or
    None then only one column is used regardless of time steps
    :param show_scaled_time: If time_group=5 then a value of 3 means t=15 if this is set to true
    :param row_labels: Group rows using none or a combination of the extras keys.
    If an empty list is passed or None, then one group is used for the whole processing unit
    :param cell_labels: Label cells using none or a combination of the extras keys.
    If an empty list is passed or None, then the cell is simply marked as has job "X" or empty "-"
    If "count" is passed then the number of jobs running in that cell is displayed
    :param show_column_labels: Whether to show column labels (time steps) or not
    :param show_row_utilization: Whether to print utilization percentage for every row (group)
    :param show_header: Whether to print a header with the unit name and the average utilization of the resource
    :param suppress_body: Whether to omit adding the body to the report. It will still be calculated which adds
    unnecessary cost to report generation. Should be changed later.
    :param long_value_handling: "trim", "wrap", "push". (Not implemented yet)
    :param cell_width: Specifies the width of each cell in the table
    :param group_name_width: Specifies the width of the first column in the table which contains the group name.
    """
    if not processing_unit.timeline_format == "stepwise":
        raise Exception("The ASCII timeline generation is currently only enabled for units which have the 'stepwise' "
                        "timeline format. Consider changing the format and trying again.")
    report = []
    duration = (processing_unit.env.now if end is None else end) - start
    if not time_grouping:
        time_grouping = int(duration)
    if not row_labels:
        row_labels = list()
    scaled_start = int(np.floor(start/time_grouping))
    scaled_end = int(np.ceil((processing_unit.env.now if end is None else end) / time_grouping))
    grouped_time_steps = range(scaled_start, scaled_end)
    total_util = 0
    # Generate groups
    groups = set()
    if len(row_labels):
        for jobs in processing_unit.timeline.values():
            for job, _ in jobs:
                values = list()
                for key in row_labels:
                    if key in job.extras.keys():
                        values.append(job.extras[key])
                    else:
                        # Add null character to signify that the key is missing
                        values.append(chr(0))
                groups.add(tuple(values))
    # Sort groups
    groups = list(groups)
    if len(groups):
        groups.sort(key=lambda x: [str(y) for y in x])
    else:
        groups.append(None)
    # Generate group rows
    rows = list()
    for group in groups:
        # Generate group name
        group_name = ""
        if group is None:
            group_name = "All"
        else:
            for i, key in enumerate(row_labels):
                group_name += "{}:{} ".format(key, group[i])
        group_name = "{{:{}}}".format(group_name_width).format(group_name)
        # Generate row body
        row_units = 0
        cells = list()
        for tg in grouped_time_steps:
            cell_units = 0
            jobs = []
            # Generate cell
            for t in range(tg*time_grouping, (tg+1) * time_grouping):
                if t in processing_unit.timeline.keys():
                    for job, units in processing_unit.timeline[t]:
                        # Does this job belong to our group ?
                        include = True
                        for i, key in enumerate(row_labels):
                            if not (key in job.extras.keys() and job.extras[key] == group[i]) and \
                                    not (key not in job.extras.keys() and group[i] == str(chr(0))):
                                include = False
                                break
                        # If it does then add it
                        if include:
                            cell_units += units
                            if job not in jobs:
                                jobs.append(job)
            row_units += cell_units
            if cell_labels is not None and cell_labels != "count" and len(cell_labels) > 0:
                jobs_text = list()
                for job in jobs:
                    values = list()
                    for key in cell_labels:
                        if key in job.extras:
                            values.append(str(job.extras[key]))
                        else:
                            values.append("X"*cell_width)
                    jobs_text.append(":".join(values))
                cells.append("{{:{}}}".format(cell_width).format(','.join(jobs_text)))
            elif cell_labels != "count":
                if len(jobs) > 0:
                    cells.append("X"*cell_width)
                else:
                    cells.append("-"*cell_width)
            else:
                cells.append("{{:<{}}}".format(cell_width).format(len(jobs)))
        if show_row_utilization:
            row_util = row_units / (processing_unit.rate * duration)
            total_util += row_util
            row_util = "{:7}".format("{:<3.2f}%".format(row_util*100))
            rows.append("|{}|{}|{}|".format(group_name, row_util, '|'.join(cells)))
        else:
            rows.append("|{}|{}|".format(group_name, '|'.join(cells)))
    # Generate header
    if show_header:
        header = "{:10} Rate: {:<5} Util: {:>6.2f}% {:5}Scheduler: {:<10}".format(str(processing_unit),
                                                                                  processing_unit.rate,
                                                                                  total_util*100, "",
                                                                                  str(processing_unit.scheduler))
        sep = "-"*max(len(rows[0]), len(header))
        report.append(sep)
        report.append(header)
        report.append(sep)
    if not suppress_body:
        # Generate column labels
        if show_column_labels:
            if show_row_utilization:
                format = "|{{:{}}}|{{:{}}}|{{:{}}}|".format(group_name_width, 7, len(rows[0])-group_name_width-11)
                report.append(format.format("Group", "Util", "Time"))
            else:
                format = "|{{:{}}}|{{:{}}}|".format(group_name_width, len(rows[0]) - group_name_width - 3)
                report.append(format.format("Group", "Time"))
            time_labels = ""
            for i in grouped_time_steps:
                time_labels += "{{:<{}}}|".format(cell_width).format(i if show_scaled_time else i*time_grouping)
            if show_row_utilization:
                report.append(format[:-1].format("","",time_labels))
            else:
                report.append(format[:-1].format("", time_labels))
        report.extend(rows)
    return '\n'.join(report)


def generate_chrome_trace_timeline(processing_unit, group_labels=None, row_labels=None, cell_labels=None,
                                   multiplier=1, utilization_bins=10, return_dict=False, display_unit="ms"):
    """
    Unit must have kept its timeline using the jobwise format
    The generated chrome trace format is documented here
    https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit?pli=1
    It can be visualized using the chrome://tracing tool.
    It is the users responsibility to choose row_labels/group_labels so that no overlapping in the cells occurs.
    Unlike the ASCII timeline, different jobs cannot share or be grouped into one cell.
    :param processing_unit: The unit to generate the trace for.
    :param group_labels: Group rows using none or a combination of the extras keys.
    Special 'unit_name' key can be passed to group based on the processing_unit
    If an empty list is passed or None, then one group is used for the all events
    :param row_labels: Group rows using none or a combination of the extras keys.
    A special 'unit_name' key can be passed to group based on the processing_unit
    If an empty list is passed or None, then one row is used for the all events
    :param cell_labels: A combination of the extras keys to use in naming the cells.
    :param utilization_bins: The number of bins of the utilization histogram. Set to None to disable.
    Setting this to a very high number can be very slow.
    :return: A json formatted string in the chrome trace format
    """
    if processing_unit.timeline is None:
        raise Exception("The given unit did not keep a timeline. A chrome trace cannot be generated.")
    # Create groups
    grouping_args = {"group_labels": group_labels, "row_labels": row_labels}
    groupings = {"group_labels": set(), "row_labels": set()}
    for key in groupings:
        for job in processing_unit.timeline:
            groupings[key].add(get_job_group(job, grouping_args[key], unit_name=processing_unit.name))
    # Add group metadata mappings
    metadata = list()
    super_group_mapping = dict()
    pid = 0
    for group in groupings["group_labels"]:
        group_name = ",".join([str(x) for x in group])
        metadata.append(dict(ph="M", name="process_name", pid=pid, args=dict(name=group_name)))
        tid = 0
        for row in groupings["row_labels"]:
            super_group_mapping[(group, row)] = (pid, tid)
            row_name = ",".join([str(x) for x in row])
            metadata.append(dict(ph="M", name="thread_name", pid=pid, tid=tid, args=dict(name=row_name)))
            tid += 1
        pid += 1
    final_pid = pid
    final_tid = tid-1
    # Add events
    events = list()
    duration_warning_flag = False
    for job in processing_unit.timeline:
        job_grouping = {"group_labels": get_job_group(job, grouping_args["group_labels"], unit_name=processing_unit.name),
                        "row_labels": get_job_group(job, grouping_args["row_labels"], unit_name=processing_unit.name)}
        pid, tid = super_group_mapping[(job_grouping["group_labels"], job_grouping["row_labels"])]
        job_name = ",".join([str(x) for x in get_job_group(job, cell_labels, unit_name=processing_unit.name)])
        for event in processing_unit.timeline[job]:
            scaled_event = dict()
            for key in event:
                scaled_event[key] = event[key] * multiplier
            if scaled_event["dur"] < 1:
                duration_warning_flag = True
            event_dict = dict(**scaled_event, name=job_name, ph="X", pid=pid, tid=tid, args=job.extras)
            events.append(event_dict)
    # if duration_warning_flag:
    #     print("Warning! the trace contains events with durations less than 1. Consider scaling your units using the "
    #           "mutliplier argument")
    # Add utilization info
    start = 0
    end = processing_unit.env.now
    total_utilization = processing_unit.get_utilization()
    if utilization_bins is not None:
        util_info = list()
        bins = list(np.arange(start, end, (end - start) / utilization_bins))
        if (end - start) % utilization_bins == 0:
            bins.append(end)
        else:
            bins[-1] = end
        bins.append(end)  # Append end again just to insert a 0 utilization at the end of the report
        for i in range(len(bins)-1):
            util = processing_unit.get_utilization(start=bins[i], end=bins[i+1])
            counter_dict = dict(pid=final_pid, name="{:.2f}%".format(total_utilization*100), ph="C",
                                ts=bins[i]*multiplier, args={"util": util})
            util_info.append(counter_dict)
        metadata.append(dict(pid=final_pid, name="process_name", ph="M",
                             args=dict(name="{}_utilization".format(processing_unit.name))))
        events.extend(util_info)
    # Concatenate and format final trace
    events.extend(metadata)
    chrome_trace = dict(traceEvents=events, final_pid=final_pid, final_tid=final_tid)
    chrome_trace["{}.{}".format(processing_unit.name, "util")] = total_utilization
    chrome_trace["displayTimeUnit"] = display_unit
    return chrome_trace if return_dict else json.dumps(chrome_trace, indent=4)


def join_chrome_traces(traces_list, sort_process_ids=True, use_trace_dict=False):
    base_trace = traces_list[0] if use_trace_dict else json.loads(traces_list[0])
    for trace in traces_list[1:]:
        if not use_trace_dict:
            trace = json.loads(trace)
        for event in trace["traceEvents"]:
            if "pid" in event:
                event["pid"] += base_trace["final_pid"] + 1
            if "tid" in event:
                event["tid"] += base_trace["final_tid"] + 1
        base_trace["traceEvents"].extend(trace["traceEvents"])
        base_trace["final_pid"] += trace["final_pid"] + 1
        base_trace["final_tid"] += trace["final_tid"] + 1
        for key in trace.keys() - {"final_pid", "final_tid", "traceEvents"}:
            base_trace[key] = trace[key]
    if sort_process_ids:
        for i in range(base_trace["final_pid"]+1):
            base_trace["traceEvents"].append(dict(ph="M", pid=i, name="process_sort_index", args=dict(sort_index=i)))
    return base_trace if use_trace_dict else json.dumps(base_trace, indent=4)


def get_gap_durations(processing_unit, group_labels=None):
    if processing_unit.timeline is not None:
        grouped_intervals = dict()
        for job, events in processing_unit.timeline.items():
            group = get_job_group(job, group_labels)
            if group not in grouped_intervals.keys():
                grouped_intervals[group] = list()
            for event in events:
                interval = (event["ts"], event["ts"] + event["dur"])
                grouped_intervals[group].append(interval)
        grouped_interval_durations = dict()
        for group, intervals in grouped_intervals.items():
            grouped_interval_durations[group] = [e-b for b, e in get_interval_gaps(intervals)]
        return grouped_interval_durations
    else:
        raise Exception("The unit must have kept a timeline dictionary to extract gap information.")
