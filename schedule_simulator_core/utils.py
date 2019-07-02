import math


class SimPrinter:
    def __init__(self, verbosity=2):
        self.verbosity = verbosity

    def print(self, env, source, msg, verbosity):
        if self.verbosity >= verbosity:
            print("t:{:<4}] {:<10}] {}".format(env.now, str(source), msg))


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
    report = []
    duration = (processing_unit.env.now if end is None else end) - start
    if not time_grouping:
        time_grouping = int(duration)
    if not row_labels:
        row_labels = list()
    scaled_start = int(math.floor(start/time_grouping))
    scaled_end = int(math.ceil((processing_unit.env.now if end is None else end) / time_grouping))
    grouped_time_steps = range(scaled_start, scaled_end)
    total_util = 0
    # Generate groups
    groups = set()
    if len(row_labels):
        for jobs in processing_unit.utilization.values():
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
                if t in processing_unit.utilization.keys():
                    for job, units in processing_unit.utilization[t]:
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


def generate_chrome_trace_timeline():
    pass
