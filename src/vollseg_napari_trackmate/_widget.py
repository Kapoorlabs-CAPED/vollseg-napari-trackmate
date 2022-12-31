"""
VollSeg Napari Track .

Made by Kapoorlabs, 2022
"""

import functools
from pathlib import Path
from typing import List, Set

import napari
import numpy as np
import pandas as pd
import seaborn as sns
from magicgui import magicgui
from magicgui import widgets as mw
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget


def plugin_wrapper_track():

    import codecs
    import xml.etree.ElementTree as et

    from csbdeep.utils import axes_check_and_normalize, axes_dict
    from napatrackmater.bTrackmate import load_tracks

    from ._data_model import pandasModel
    from ._table_widget import TrackTable

    DEBUG = False

    def _raise(e):
        if isinstance(e, BaseException):
            raise e
        else:
            raise ValueError(e)

    def get_data(image, debug=DEBUG):

        image = image.data[0] if image.multiscale else image.data
        if debug:
            print("image loaded")
        return np.asarray(image)

    def get_xml_data(xml_path):

        root = et.fromstring(codecs.open(xml_path, "r", "utf8").read())

        filtered_track_ids = [
            int(track.get("TRACK_ID"))
            for track in root.find("Model")
            .find("FilteredTracks")
            .findall("TrackID")
        ]

        # Extract the tracks from xml
        tracks = root.find("Model").find("AllTracks")

        x_ls = tracks.findall("Track")
        print("total tracks", len(x_ls))

        for track in x_ls:

            track_id = int(track.get("TRACK_ID"))

            if track_id in filtered_track_ids:
                (
                    tracklets,
                    DividingTrajectory,
                    split_points_times,
                ) = load_tracks(track, track_id, filtered_track_ids)
        return (
            root,
            filtered_track_ids,
            tracks,
            tracklets,
            DividingTrajectory,
            split_points_times,
        )

    def get_label_data(image, debug=DEBUG):

        image = image.data[0] if image.multiscale else image.data
        if debug:
            print("Label image loaded")
        return np.asarray(image).astype(np.uint16)

    DEFAULTS_COLOR = dict(
        track_atribute_color="NUMBER_SPLITS", spot_attribute_color="QUALITY"
    )

    def abspath(root, relpath):
        root = Path(root)
        if root.is_dir():
            path = root / relpath
        else:
            path = root.parent / relpath
        return str(path.absolute())

    def change_handler(*widgets, init=False, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:
                    print(f"{str(emitter.name).upper()}: {source.name}")
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

    worker = None

    DEFAULTS_MODEL = dict(axes="TZYX")

    DEFAULTS_FUNC_PARAMETERS = dict()

    @magicgui(
        defaults_params_button=dict(
            widget_type="PushButton", text="Restore Parameter Defaults"
        ),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        layout="vertical",
        persist=False,
        call_button=False,
    )
    def plugin_function_parameters(
        defaults_params_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        return plugin_function_parameters

    @magicgui(
        spot_attribute=dict(
            widget_type="ComboBox",
            visible=False,
            label="Spot Attributes",
            value=DEFAULTS_COLOR["spot_attribute_color"],
        ),
        track_attribute=dict(
            widget_type="ComboBox",
            visible=False,
            label="Track Attributes",
            value=DEFAULTS_COLOR["track_atribute_color"],
        ),
    )
    def plugin_color_parameters(
        spot_attribute,
        track_attribute,
        persist=True,
        call_button=True,
    ) -> List[napari.types.LayerDataTuple]:

        return plugin_color_parameters

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.25080/majora-1b6fd038-014")

    @magicgui(
        label_head=dict(
            widget_type="Label",
            label=f'<h1> <img src="{kapoorlogo}"> </h1>',
            value=f'<h5><a href=" {citation}"> NapaTrackMater: Track Analysis of TrackMate in Napari</a></h5>',
        ),
        image=dict(label="Input Image"),
        seg_image=dict(label="Optional Segmentation Image"),
        mask_image=dict(label="Optional Mask Image"),
        xml_path=dict(
            widget_type="FileEdit",
            visible=False,
            label="TrackMate xml",
            mode="f",
        ),
        track_csv=dict(
            widget_type="FileEdit", visible=False, label="Track csv", mode="f"
        ),
        spot_csv=dict(
            widget_type="FileEdit", visible=False, label="Spot csv", mode="f"
        ),
        edges_csv=dict(
            widget_type="FileEdit",
            visible=False,
            label="Edges/Links csv",
            mode="f",
        ),
        axes=dict(
            widget_type="LineEdit",
            label="Image Axes",
            value=DEFAULTS_MODEL["axes"],
        ),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        layout="vertical",
        persist=True,
        call_button=True,
    )
    def plugin(
        viewer: napari.Viewer,
        label_head,
        image: napari.layers.Image,
        seg_image: napari.layers.Labels,
        mask_image: napari.layers.Labels,
        xml_path,
        track_csv,
        spot_csv,
        edges_csv,
        axes,
        n_tiles,
        defaults_model_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        x = get_data(image)
        print(x.shape)
        x_seg = get_label_data(seg_image)
        x_mask = get_label_data(mask_image)
        print(x_seg.path, x_mask.path)

        (
            root,
            filtered_track_ids,
            tracks,
            tracklets,
            DividingTrajectory,
            split_points_times,
        ) = get_xml_data(xml_path)

        axes = axes_check_and_normalize(axes, length=x.ndim)
        nonlocal worker
        progress_bar.label = "Starting TrackMate analysis"

        if "T" in axes:
            t = axes_dict(axes)["T"]
            n_frames = x.shape[t]

            def progress_thread(current_time):

                progress_bar.label = "Fitting Functions (files)"
                progress_bar.range = (0, n_frames - 1)
                progress_bar.value = current_time
                progress_bar.show()

        if "T" in axes:
            # determine scale for output axes
            worker.returned.connect()
            worker.yielded.connect()

        else:
            worker = ()
            worker.returned.connect()

        progress_bar.hide()

    plugin.label_head.value = '<br>Citation <tt><a href="https://doi.org/10.25080/majora-1b6fd038-014" style="color:gray;">NapaTrackMater Scipy</a></tt>'
    plugin.label_head.native.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
    )

    tabs = QTabWidget()

    parameter_function_tab = QWidget()
    _parameter_function_tab_layout = QVBoxLayout()
    parameter_function_tab.setLayout(_parameter_function_tab_layout)
    _parameter_function_tab_layout.addWidget(plugin_function_parameters.native)
    tabs.addTab(parameter_function_tab, "Parameter Selection")

    color_tracks_tab = QWidget()
    _color_tracks_tab_layout = QVBoxLayout()
    color_tracks_tab.setLayout(_color_tracks_tab_layout)
    _color_tracks_tab_layout.addWidget()
    tabs.addTab(color_tracks_tab, "Color Tracks")

    canvas = FigureCanvas()
    canvas.figure.set_tight_layout(True)
    ax = canvas.figure.subplots(2, 2)

    plot_tab = canvas
    _plot_tab_layout = QVBoxLayout()
    plot_tab.setLayout(_plot_tab_layout)
    _plot_tab_layout.addWidget(plot_tab)
    tabs.addTab(plot_tab, "Plots")

    table_tab = TrackTable()
    _table_tab_layout = QVBoxLayout()
    table_tab.setLayout(_table_tab_layout)
    _table_tab_layout.addWidget(table_tab)
    tabs.addTab(table_tab, "Table")

    plugin.native.layout().addWidget(tabs)

    def _selectInTable(selected_data: Set[int]):
        """Select in table in response to viewer (add, highlight).

        Args:
            selected_data (set[int]): Set of selected rows to select
        """

        table_tab.mySelectRows(selected_data)

    def _slot_data_change(
        action: str, selection: set, layerSelectionCopy: dict
    ):

        df = table_tab.myModel._data

        if action == "select":
            # TODO (cudmore) if Layer is labaeled then selection is a list
            if isinstance(selection, list):
                selection = set(selection)
            _selectInTable(selection)
            table_tab.signalDataChanged.emit(action, selection, df)

        elif action == "add":
            # addedRowList = selection
            # myTableData = self.getLayerDataFrame(rowList=addedRowList)
            myTableData = df
            table_tab.myModel.myAppendRow(myTableData)
            _selectInTable(selection)
            table_tab.signalDataChanged.emit(action, selection, df)
        elif action == "delete":
            # was this
            deleteRowSet = selection
            # logger.info(f'myEventType:{myEventType} deleteRowSet:{deleteRowSet}')
            # deletedDataFrame = self.myTable2.myModel.myGetData().iloc[list(deleteRowSet)]

            _deleteRows(deleteRowSet)

            # self._blockDeleteFromTable = True
            # self.myTable2.myModel.myDeleteRows(deleteRowList)
            # self._blockDeleteFromTable = False

            table_tab.signalDataChanged.emit(action, selection, df)
        elif action == "change":
            moveRowList = list(selection)  # rowList is actually indexes
            myTableData = df
            # myTableData = self.getLayerDataFrame(rowList=moveRowList)
            table_tab.myModel.mySetRow(moveRowList, myTableData)

            table_tab.signalDataChanged.emit(action, selection, df)

    def _slot_selection_changed(selectedRowList: List[int], isAlt: bool):
        """Respond to user selecting a table row.
        Note:
            - This is coming from user selection in table,
                we do not want to propogate
        """

        df = table_tab.myModel._data
        # selectedRowSet = set(selectedRowList)

        print(df)

        # table_tab.signalDataChanged.emit("select", selectedRowSet, df)

    def _deleteRows(rows: Set[int]):
        table_tab.myModel.myDeleteRows(rows)

    def _refreshPlotData(df):

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].cla()

        sns.violinplot(x="Plot_Name", data=df, ax=ax[0, 0])

        ax[0, 0].set_xlabel("Plot Name")

        canvas.draw()

    def _refreshTableData(df: pd.DataFrame):
        """Refresh all data in table by setting its data model from provided dataframe.
        Args:
            df (pd.DataFrame): Pandas dataframe to refresh with.
        """

        if table_tab is None:
            # interface has not been initialized
            return

        if df is None:
            return
        TrackModel = pandasModel(df)
        table_tab.mySetModel(TrackModel)
        _refreshPlotData(df)

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet(
                "" if valid else "background-color: red"
            )

    table_tab.signalDataChanged.connect(_slot_data_change)
    table_tab.signalSelectionChanged.connect(_slot_selection_changed)

    @change_handler(plugin_function_parameters.defaults_params_button)
    def restore_function_parameters_defaults():
        for k, v in DEFAULTS_FUNC_PARAMETERS.items():
            getattr(plugin_function_parameters, k).value = v

    # -> triggered by napari (if there are any open images on plugin launch)

    def function_calculator(ndim: int):

        data = []

        df = pd.DataFrame(
            data,
            columns=[],
        )
        _refreshTableData(df)

    @change_handler(plugin.image, init=False)
    def _image_change(image: napari.layers.Image):
        plugin.image.tooltip = (
            f"Shape: {get_data(image).shape, str(image.name)}"
        )

        # dimensionality of selected model: 2, 3, or None (unknown)

        axes = None

        axes = "TYX"
        plugin.recompute_current_button.show()

        if axes == plugin.axes.value:
            # make sure to trigger a changed event, even if value didn't actually change
            plugin.axes.changed(axes)
        else:
            plugin.axes.value = axes

    # -> triggered by _image_change
    @change_handler(plugin.axes, init=False)
    def _axes_change():
        value = plugin.axes.value
        print(f"axes is {value}")

    return plugin
