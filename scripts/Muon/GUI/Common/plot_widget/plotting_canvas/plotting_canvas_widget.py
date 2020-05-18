# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2020 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +

from MultiPlotting.QuickEdit.quickEdit_widget import QuickEditWidget
from Muon.GUI.Common.plot_widget.plotting_canvas.plotting_canvas_model import PlottingCanvasModel
from Muon.GUI.Common.plot_widget.plotting_canvas.plotting_canvas_presenter import PlottingCanvasPresenter
from Muon.GUI.Common.plot_widget.plotting_canvas.plotting_canvas_view import PlottingCanvasView


class PlottingCanvasWidget(object):

    def __init__(self, parent, context):

        self._figure_options = QuickEditWidget(parent)
        self._plotting_view = PlottingCanvasView(parent)
        self._plotting_view.add_widget(self._figure_options.widget)
        self._model = PlottingCanvasModel(context)
        self._presenter = PlottingCanvasPresenter(self._plotting_view, self._model, self._figure_options)

    @property
    def presenter(self):
        return self._presenter

    @property
    def widget(self):
        return self._plotting_view
