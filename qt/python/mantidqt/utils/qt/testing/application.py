# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2017 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
#  This file is part of the mantid workbench.
from __future__ import absolute_import

import atexit

from qtpy.QtWidgets import QApplication

from mantidqt.utils.qt.plugins import setup_library_paths

# Hold on to QAPP reference to avoid garbage collection
_QAPP = QApplication.instance()


@atexit.register
def cleanup_qapp_ref():
    """
    Remove the global reference to the QApplication object here
    """
    global _QAPP
    del _QAPP


def get_application(name=''):
    """
    Initialise and return the global application object
    :param name: Optional application name
    :return: Global appliction object
    """
    global _QAPP
    if _QAPP is None:
        setup_library_paths()
        _QAPP = QApplication([name])
    return _QAPP
