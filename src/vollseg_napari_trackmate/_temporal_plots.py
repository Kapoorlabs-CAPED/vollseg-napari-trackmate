from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from qtpy import QtCore
from qtpy.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget


class TemporalStatistics(QWidget):
    def __init__(self, tabs, parent=None, min_height=500, min_width=500):

        super().__init__()
        self.tabs = tabs
        self.min_height = min_height
        self.min_width = min_width
        self._set_model()

    def _set_model(self):

        self.stat_plot_tab = QWidget()
        self.scroll_area = QScrollArea()
        self.scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOn
        )
        self.scroll_container = QWidget()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_container)
        self.scroll_layout = QHBoxLayout(self.scroll_container)
        self.lay = QVBoxLayout(self.stat_plot_tab)
        self.lay.addWidget(self.scroll_area)
        self.container = None

    def _repeat_after_plot(self):

        self.stat_canvas = FigureCanvas(Figure())
        self.stat_ax = self.stat_canvas.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.stat_canvas, self.tabs)
        self.container = QWidget()
        self.lay = QVBoxLayout(self.container)
        self.lay.addWidget(self.stat_canvas)
        self.lay.addWidget(self.toolbar)
        self.scroll_layout.addWidget(self.container)
        self.container.setMinimumWidth(self.min_width)
        self.container.setMinimumHeight(self.min_height)
        self.stat_canvas.draw()

    def _reset_container(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self._reset_container(child.layout())
            self.container = layout
