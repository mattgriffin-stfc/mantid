# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2020 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
#  This file is part of the mantid workbench.

# 3rd party imports
from qtpy.QtWidgets import QGroupBox, QVBoxLayout, QWidget

# local imports
from mantidqt.widgets.workspacedisplay.table.view import TableWorkspaceDisplayView, QTableWidget


class PeaksWorkspaceTableView(TableWorkspaceDisplayView):
    """Specialization of a table view to display peaks"""

    def keyPressEvent(self, event):
        QTableWidget.keyPressEvent(self, event)


class PeaksViewerView(QWidget):
    """Displays a table view of the PeaksWorkspace along with controls
    to interact with the peaks.
    """
    TITLE_PREFIX = "Workspace: "

    def __init__(self, painter, sliceinfo_provider, parent=None):
        """
        :param painter: An object responsible for draw the peaks representations
        :param sliceinfo_provider: An object responsible for providing access to current slice information
        :param parent: An optional parent widget
        """
        super().__init__(parent)
        self._painter = painter
        self._sliceinfo_provider = sliceinfo_provider
        self._current_selection = None
        self._group_box = None
        self._presenter = None
        self._table_view = None
        self._setup_ui()

    @property
    def painter(self):
        """Return a reference to the painter used in this view"""
        return self._painter

    @property
    def sliceinfo(self):
        """Return information regarding the current slice"""
        return self._sliceinfo_provider.get_sliceinfo()

    @property
    def table_view(self):
        return self._table_view

    @property
    def selected_index(self):
        # cache current selection for checking in mouse click handler
        self._current_selection = self._selected_index()
        return self._current_selection

    def set_peak_color(self, peak_color):
        """
        Set the color of the peak represented in this view
        :param peak_color: A str describing the color of the displayed peak
        """
        self._group_box.setStyleSheet(f"QGroupBox  {{color: {peak_color}}};")

    def set_slicepoint(self, value):
        """
        Set the slice point to the given value
        :param value: Float giving the current slice value
        """
        self._sliceinfo_provider.set_slicevalue(value)

    def set_title(self, name):
        """
        :param name: Set the name label for the workspace
        """
        self._group_box.setTitle(self.TITLE_PREFIX + name)

    def subscribe(self, presenter):
        """
        :param presenter: An object to handle GUI events.
        """
        self._presenter = presenter
        self._table_view.subscribe(presenter)

    # private api
    def _setup_ui(self):
        """
        Arrange the widgets on the window
        """
        self._group_box = QGroupBox(self)
        self._group_box.setContentsMargins(0, 0, 0, 0)
        self._table_view = PeaksWorkspaceTableView(parent=self)
        self._table_view.setSelectionBehavior(PeaksWorkspaceTableView.SelectRows)
        self._table_view.setSelectionMode(PeaksWorkspaceTableView.SingleSelection)
        # itemSelectionChanges handles selection changes from either keyboard/mouse
        self._table_view.itemSelectionChanged.connect(self._on_row_selection_changed)
        # the selection might not change when an item is clicked but we want to notify
        # the outside world
        self._table_view.itemClicked.connect(self._on_row_clicked)

        group_box_layout = QVBoxLayout()
        group_box_layout.addWidget(self._table_view)
        self._group_box.setLayout(group_box_layout)
        widget_layout = QVBoxLayout()
        widget_layout.addWidget(self._group_box)
        self.setLayout(widget_layout)

    def _on_row_selection_changed(self):
        """
        Notify that a different peak has been selected. It is assumed only single row selection is allowed
        """
        self._presenter.notify(self._presenter.Event.PeakSelected)

    def _on_row_clicked(self, _):
        """
        When a peak is clicked check if it is already selected and notify that this
        peak has been selected again. Handles the case when selecting the same
        peak needs to reset another view. Care is taken to avoid emitting the peak
        selection notification when peak selection changes as _on_row_selection_changed
        handles this.
        """
        if self._current_selection == self._selected_index():
            self._presenter.notify(self._presenter.Event.PeakSelected)

    def _selected_index(self):
        # construction ensures we can only have 0 or 1 items selected
        selected = self.table_view.selectedItems()
        if not selected:
            return None

        return self.table_view.row(selected[0])


class PeaksViewerCollectionView(QWidget):
    """Display a collection of PeaksViewerView objects in a scrolling view.
    """

    def __init__(self, painter, sliceinfo_provider, parent=None):
        """
        :param painter: An object responsible for draw the peaks representations
        :param sliceinfo_provider: An object responsible for providing access to current slice information
        :param parent: An optional parent widget
        """
        super(PeaksViewerCollectionView, self).__init__(parent)
        self._painter = painter
        self._sliceinfo_provider = sliceinfo_provider
        self._setup_ui()

    def append_peaksviewer(self):
        """
        Append a view widget to end of the current list of views
        :param peaks_view: A reference to a single view widget for a PeaksWorkspace
        """
        child_view = PeaksViewerView(self._painter, self._sliceinfo_provider, parent=self)
        self._peaks_layout.addWidget(child_view)
        return child_view

    def remove_peaksviewer(self, widget):
        """
        Remove a PeaksViewer from the collection
        :param widget: A reference to the PeaksViewerView
        """
        layout = self._peaks_layout
        item = layout.takeAt(layout.indexOf(widget))
        item.widget().deleteLater()

    # private api
    def _setup_ui(self):
        """
        Arrange widgets on the window.
        """
        # create vertical layouts for outer widget
        outer_layout = QVBoxLayout()
        self.setLayout(outer_layout)
        self._peaks_layout = outer_layout
