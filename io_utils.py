import math
class SimPrinter:
    def __init__(self, verbosity=2):
        self.verbosity = verbosity

    def print(self, env, source, msg, verbosity):
        if self.verbosity >= verbosity:
            print("t:{:<4}] {}] {}".format(env.now, source, msg))


def generate_report(processing_unit, start=0, end=None,
                    time_grouping=1, show_scaled_time=False,
                    row_labels=None, cell_labels=None,
                    show_column_labels=True, show_row_utilization=True, show_header=True,
                    long_value_handling="trim"):
    """
    :param start: From which time step should we generate the report (Inclusive)
    :param end: To which time step (Inclusive)
    :param time_grouping: How many time steps per column. If set to 0 or
    None then only one column is used regardless of time steps
    :param show_scaled_time: If time_group=5 then a value of 3 means t=15 if this is set to true
    :param row_labels: Group rows using none or a combination of the extras keys.
    If an empty list is passed or None, then one group is used for the whole processing unit
    :param cell_labels: Label cells using none or a combination of the extras keys.
    If an empty list is passed or None, then the cell is simply marked as has job "X" or empty "-"
    :param show_column_labels: Whether to show column labels (time steps) or not
    :param show_row_utilization: Whether to print utilization percentage for every row (group)
    :param show_header: Whether to print a header with the unit name and the average utilization of the resource
    :param long_value_handling: "trim", "wrap", "push"
    """
    cell_width = 5
    row_header_width = 30
    report = []
    duration = (processing_unit.env.now if end is None else end) - start
    if not time_grouping:
        time_scale = duration
    scaled_start = int(math.floor(start/time_grouping))
    scaled_end = int(math.ceil((processing_unit.env.now if end is None else end) / time_grouping))
    grouped_time_steps = range(scaled_start, scaled_end)
    avg_util = 0
    # Generate header
    if show_header:
        report.append("{} Util: {:<3.2f}% -------------------------------------------".format(processing_unit, avg_util * 100))
    # Generate column labels
    if show_column_labels:
        if show_scaled_time:
            column_labels = "t({{:<{}}})|".format(row_header_width-3).format(time_grouping)
        else:
            column_labels = "t{{:<{}}}|".format(row_header_width-1).format("")
        for i in grouped_time_steps:
            column_labels += "{{:<{}}}|".format(cell_width).format(i if show_scaled_time else i*time_grouping)
        report.append(column_labels)
    # Generate groups
    groups = set()
    if row_labels is not None and len(row_labels) > 0:
        for jobs in processing_unit.utilization.values():
            for job, _ in jobs:
                values = list()
                for key in row_labels:
                    # We add a true/false flag before the value so that the sorting algorithm can sort None values
                    if key in job.extras.keys():
                        values.append(job.extras[key])
                    else:
                        # Add null character to signify that the key is missing
                        values.append(chr(0))
                groups.add(tuple(values))
    # Sort groups
    groups = list(groups)
    groups.sort(key=lambda x: [str(y) for y in x])
    # Generate group rows
    for group in groups:
        # Generate row header
        row_header = ""
        for i, key in enumerate(row_labels):
            row_header += "{}:{} ".format(key, group[i])
        # Generate row body
        row_units = 0
        cells = list()
        for tg in grouped_time_steps:
            cell_units = 0
            jobs = []
            # Generate cell
            for t in range(tg,tg+time_grouping):
                if t in processing_unit.utilization.keys():
                    for job, units in processing_unit.utilization[t]:
                        include = True
                        for i, key in enumerate(row_labels):
                            if not (key in job.extras.keys() and job.extras[key] == group[i]) and \
                                    not (key not in job.extras.keys() and group[i] == str(chr(0))):
                                include = False
                                break

                        if include:
                            cell_units += units
                            jobs.append(job)
            row_units += cell_units
            if cell_labels is not None and len(cell_labels) > 0:
                jobs_text = list()
                for job in jobs:
                    values = list()
                    for key in cell_labels:
                        if key in job.extras:
                            values.append(str(job.extras[key]))
                        else:
                            values.append("")
                    jobs_text.append(":".join(values))
                cells.append("{{:{}}}".format(cell_width).format(','.join(jobs_text)))
            else:
                if len(jobs) > 0:
                    cells.append("X"*cell_width)
                else:
                    cells.append("-"*cell_width)
        row_util = row_units / (processing_unit.rate * duration)
        if show_row_utilization:
            row_header += "({:<3.2f}%) ".format(row_util*100)
        row_header = "{{:{}}}|".format(row_header_width).format(row_header)
        report.append(row_header + '|'.join(cells) + '|')
    return '\n'.join(report)