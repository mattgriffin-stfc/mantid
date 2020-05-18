# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
#  This file is part of the mantid workbench.
#
#
from mantid.kernel import (BoolTimeSeriesProperty, BoolFilteredTimeSeriesProperty,
                           FloatTimeSeriesProperty, FloatFilteredTimeSeriesProperty,
                           Int32TimeSeriesProperty, Int32FilteredTimeSeriesProperty,
                           Int64TimeSeriesProperty, Int64FilteredTimeSeriesProperty,
                           StringTimeSeriesProperty, StringFilteredTimeSeriesProperty,
                           logger)
from mantid.api import MultipleExperimentInfos
from qtpy.QtGui import QStandardItemModel, QStandardItem

TimeSeriesProperties = (BoolTimeSeriesProperty,
                        FloatTimeSeriesProperty, Int32TimeSeriesProperty,
                        Int64TimeSeriesProperty, StringTimeSeriesProperty)
FilteredTimeSeriesProperties = (BoolFilteredTimeSeriesProperty,
                                FloatFilteredTimeSeriesProperty, Int32FilteredTimeSeriesProperty,
                                Int64FilteredTimeSeriesProperty, StringFilteredTimeSeriesProperty)


def get_type(log):
    """Convert type to something readable"""
    dtype_map = {'i': 'int', 'f': 'float', 's': 'string', 'b': 'bool'}
    if isinstance(log, TimeSeriesProperties):
        return "{} series".format(dtype_map[log.dtype()[0].lower()])
    else:
        return log.type


def get_value(log):
    """Returns the either the value or the number of entries
    """
    MAX_LOG_SIZE = 20  # the maximum log length to try to show in the value column

    if isinstance(log, TimeSeriesProperties):
        if log.size() == 1:
            return '{} (1 entry)'.format(log.firstValue())
        else:
            entry_descr = '({} entries)'.format(log.size())

            # show the value if they are all the same
            if log.size() < MAX_LOG_SIZE:
                value = set(log.value)
                if len(value) == 1:
                    return "{} {}".format(value.pop(), entry_descr)

            # otherwise just show the number of values
            return entry_descr
    else:
        return log.value


class SampleLogsModel(object):
    """This class stores the workspace object and return log values when
    requested
    """

    def __init__(self, ws):
        """Stores three thing:, the workspace, which experiment info number
        to use, and the run object.
        """
        self._ws = ws
        self._exp = 0
        self._set_run()

    def _set_run(self):
        """Set run depending on workspace type and experiment info number"""
        if self.isMD():
            self.run = self._ws.getExperimentInfo(self._exp).run()
        else:
            self.run = self._ws.run()

    def set_exp(self, exp):
        """Change the experiment info number"""
        self._exp = exp
        self._set_run()

    def get_exp(self):
        """Return the experiment info number"""
        return self._exp

    def get_ws(self):
        """Return the workspace"""
        return self._ws

    def get_name(self):
        """Return the workspace name"""
        return self._ws.name()

    def getNumExperimentInfo(self):
        """Return number of experiment info's in workspace"""
        return self._ws.getNumExperimentInfo() if self.isMD() else 0

    def get_log(self, LogName):
        """Return log of given LogName"""
        return self.run.getLogData(LogName)

    def get_is_log_filtered(self, LogName):
        """Return if the log of given LogName is filtered"""
        log = self.get_log(LogName)
        return isinstance(log, FilteredTimeSeriesProperties)

    def get_log_names(self):
        """Returns a list of logs in workspace"""
        return self.run.keys()

    def get_log_display_values(self, LogName):
        """Return a row to display for a log (name, type, value, units)"""
        log = self.get_log(LogName)
        return log.name, get_type(log), get_value(log), log.units

    def are_any_logs_plottable(self):
        """returns true if any of the logs are plottable.
        Only Float, Int32 and Int64
        TimeSeriesProperties are plottable at this point.
        """
        log_names = self.get_log_names()
        for log_name in log_names:
            if self.is_log_plottable(log_name):
                return True
        return False

    def is_log_plottable(self, LogName):
        """Checks if logs is plottable. Only Float, Int32 and Int64
        TimeSeriesProperties are plottable at this point.
        """
        return isinstance(self.get_log(LogName), (FloatTimeSeriesProperty,
                                                  Int32TimeSeriesProperty,
                                                  Int64TimeSeriesProperty))

    def get_statistics(self, LogName, filtered = True):
        """Return the statistics of a particular log"""
        log = self.get_log(LogName)
        if isinstance(log, TimeSeriesProperties):
            if ((not filtered) and isinstance(log,FilteredTimeSeriesProperties)):
                log = log.unfiltered()
            return log.getStatistics()

    def isMD(self):
        """Checks if workspace is a MD Workspace"""
        return isinstance(self._ws, MultipleExperimentInfos)

    def getItemModel(self):
        """Return a QModel made from the current workspace. This should be set
        onto a QTableView
        """

        def create_table_item(column, itemname, callable, *args):
            item = QStandardItem()
            item.setEditable(False)
            try:
                item.setText(callable(*args))
            except Exception as exc:
                logger.warning("Error setting column {} for log {}: {}".format(column, itemname, str(exc)))

            return item

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Name", "Type", "Value", "Units"])
        model.setColumnCount(4)
        for key in self.get_log_names():
            log = self.run.getLogData(key)
            name = create_table_item("Name", key, lambda: log.name)
            log_type = create_table_item("Type", key, get_type, log)
            value = create_table_item("Value", key, lambda log: str(get_value(log)), log)
            unit = create_table_item("Units", key, lambda: log.units)
            model.appendRow((name, log_type, value, unit))

        model.sort(0)
        return model
