from __future__ import absolute_import, print_function


from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QHBoxLayout, QProgressBar, QPushButton


from .presenter import AlgorithmProgressPresenter


class AlgorithmProgressWidget(QWidget):

    def __init__(self, parent=None):
        super(AlgorithmProgressWidget, self).__init__(parent)
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignHCenter)
        self.progress_bar.setValue(5)
        self.details_button = QPushButton('Details')
        layout = QHBoxLayout()
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.details_button)
        self.setLayout(layout)
        self.presenter = AlgorithmProgressPresenter(self)
