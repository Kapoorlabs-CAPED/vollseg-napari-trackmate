"""
VollSeg Napari Track .

Made by Kapoorlabs, 2022
"""

import functools
import math
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
from tqdm import tqdm


def plugin_wrapper_track():

    import codecs
    import xml.etree.ElementTree as et

    from csbdeep.utils import axes_check_and_normalize, axes_dict
    from napari.qt.threading import thread_worker
    from napatrackmater.bTrackmate import load_tracks, normalizeZeroOne
    from skimage.util import map_array

    from ._data_model import pandasModel
    from ._table_widget import TrackTable

    DEBUG = False
    scale = 255 * 255
    # Boxname = "TrackBox"
    AttributeBoxname = "AttributeIDBox"
    TrackAttributeBoxname = "TrackAttributeIDBox"

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

    def Relabel(image, locations):

        print("Relabelling image with chosen trackmate attribute")
        Newseg_image = np.copy(image)
        for p in tqdm(range(0, Newseg_image.shape[0])):

            sliceimage = Newseg_image[p, :]
            originallabels = []
            newlabels = []
            for relabelval, centroid in locations:
                if len(Newseg_image.shape) == 4:
                    time, z, y, x = centroid
                else:
                    time, y, x = centroid

                if p == int(time):

                    if len(Newseg_image.shape) == 4:
                        originallabel = sliceimage[z, y, x]
                    else:
                        originallabel = sliceimage[y, x]

                    if originallabel == 0:
                        relabelval = 0
                    if math.isnan(relabelval):
                        relabelval = -1
                    originallabels.append(int(originallabel))
                    newlabels.append(int(relabelval))

            relabeled = map_array(
                sliceimage, np.asarray(originallabels), np.asarray(newlabels)
            )
            Newseg_image[p, :] = relabeled

        return Newseg_image

    def get_xml_data(xml_path):

        root = et.fromstring(codecs.open(xml_path, "r", "utf8").read())

        nonlocal xcalibration, ycalibration, zcalibration, tcalibration

        filtered_track_ids = [
            int(track.get("TRACK_ID"))
            for track in root.find("Model")
            .find("FilteredTracks")
            .findall("TrackID")
        ]

        # Extract the tracks from xml
        tracks = root.find("Model").find("AllTracks")
        settings = root.find("Settings").find("ImageData")

        xcalibration = float(settings.get("pixelwidth"))
        ycalibration = float(settings.get("pixelheight"))
        zcalibration = float(settings.get("voxeldepth"))
        tcalibration = int(float(settings.get("timeinterval")))

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

    def get_csv_data(csv):

        dataset = pd.read_csv(csv, delimiter=",")[3:]
        dataset_index = dataset.index

        return dataset, dataset_index

    def get_track_dataset(track_dataset, track_dataset_index):

        nonlocal AllTrackValues, AllTrackKeys
        AllTrackValues = []
        AllTrackKeys = []

        for k in track_dataset.keys():
            if k == "TRACK_ID":
                Track_id = track_dataset[k].astype("float")
                indices = np.where(Track_id == 0)
                maxtrack_id = max(Track_id)
                condition_indices = track_dataset_index[indices]
                Track_id[condition_indices] = maxtrack_id + 1
                AllTrackValues.append(Track_id)
                AllTrackKeys.append(k)
            else:
                x = track_dataset[k].astype("float")
                minval = min(x)
                maxval = max(x)

                if minval > 0 and maxval <= 1:

                    x = normalizeZeroOne(x, scale=scale)

                AllTrackKeys.append(k)
                AllTrackValues.append(x)

        TrackAttributeids = []
        for attributename in AllTrackKeys:
            TrackAttributeids.append(attributename)
        for i in range(0, len(TrackAttributeids)):
            plugin_color_parameters.track_attributes.addItem(
                str(TrackAttributeids[i])
            )

    def get_spot_dataset(spot_dataset, spot_dataset_index):

        nonlocal AllKeys
        AllKeys = []

        for k in spot_dataset.keys():

            if k == "TRACK_ID":
                Track_id = spot_dataset[k].astype("float")
                indices = np.where(Track_id == 0)
                maxtrack_id = max(Track_id)
                condition_indices = spot_dataset_index[indices]
                Track_id[condition_indices] = maxtrack_id + 1
                AllValues.append(Track_id)

            if k == "POSITION_X":
                LocationX = (
                    spot_dataset["POSITION_X"].astype("float") / xcalibration
                ).astype("int")
                AllValues.append(LocationX)

            if k == "POSITION_Y":
                LocationY = (
                    spot_dataset["POSITION_Y"].astype("float") / ycalibration
                ).astype("int")
                AllValues.append(LocationY)
            if k == "POSITION_Z":
                LocationZ = (
                    spot_dataset["POSITION_Z"].astype("float") / zcalibration
                ).astype("int")
                AllValues.append(LocationZ)
            if k == "FRAME":
                LocationT = (spot_dataset["FRAME"].astype("float")).astype(
                    "int"
                )
                AllValues.append(LocationT)
            elif (
                k != "TRACK_ID"
                and k != "POSITION_X"
                and k != "POSITION_Y"
                and k != "POSITION_Z"
                and k != "FRAME"
            ):
                AllValues.append(spot_dataset[k].astype("float"))

            AllKeys.append(k)

        Attributeids = []
        for attributename in AllKeys:
            Attributeids.append(attributename)
        for i in range(0, len(Attributeids)):
            plugin_color_parameters.spot_attributes.addItem(
                str(Attributeids[i])
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

    AllTrackValues = None
    # AllTrackID = None
    AllTrackKeys = None
    # AllTrackAttr = None
    AllValues = None
    AllKeys = None
    xcalibration = None
    ycalibration = None
    zcalibration = None
    tcalibration = None

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
        spot_attributes=dict(
            widget_type="ComboBox",
            visible=True,
            label="Spot Attributes",
        ),
        track_attributes=dict(
            widget_type="ComboBox",
            visible=True,
            label="Track Attributes",
        ),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        persist=True,
        call_button=True,
    )
    def plugin_color_parameters(
        spot_attributes,
        track_attributes,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        nonlocal worker

        spot_attributes.addItem(AttributeBoxname)
        track_attributes.addItem(TrackAttributeBoxname)

        worker = _Color_tracks(spot_attributes.value, track_attributes.value)
        worker.returned.connect(return_color_tracks)
        if "T" in plugin.axes.value:
            t = axes_dict(plugin.axes.value)["T"]
            n_frames = plugin.image.value.shape[t]

            def progress_thread(current_time):

                progress_bar.label = "Coloring cells with chosen attribute"
                progress_bar.range = (0, n_frames - 1)
                progress_bar.value = current_time
                progress_bar.show()

            worker.yielded.connect(return_color_tracks)

        return plugin_color_parameters

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.25080/majora-1b6fd038-014")

    def return_color_tracks(new_seg_image, attribute):

        plugin.viewer.value.add_labels(new_seg_image, name=attribute)

    @thread_worker(connect={"returned": return_color_tracks})
    def _Color_tracks(spot_attribute, track_attribute):

        yield 0

        if spot_attribute is not None and AllKeys is not None:

            attribute = spot_attribute
            for k in range(len(AllKeys)):

                if AllKeys[k] == "POSITION_X":
                    keyX = k
                if AllKeys[k] == "POSITION_Y":
                    keyY = k
                if AllKeys[k] == "POSITION_Z":
                    keyZ = k
                if AllKeys[k] == "FRAME":
                    keyT = k

            for count, k in enumerate(range(len(AllKeys))):
                yield count
                locations = []
                if AllKeys[k] == spot_attribute:

                    for attr, time, z, y, x in tqdm(
                        zip(
                            AllValues[k],
                            AllValues[keyT],
                            AllValues[keyZ],
                            AllValues[keyY],
                            AllValues[keyX],
                        ),
                        total=len(AllValues[k]),
                    ):
                        if len(plugin.seg_image.value.shape) == 4:
                            centroid = (time, z, y, x)
                        else:
                            centroid = (time, y, x)

                        locations.append([attr, centroid])

        if (
            track_attribute is not None
            and AllTrackKeys is not None
            and AllKeys is not None
        ):

            attribute = track_attribute
            idattr = {}
            for k in range(len(AllTrackKeys)):

                if AllTrackKeys[k] == "TRACK_ID":
                    p = k

            for k in range(len(AllTrackKeys)):

                if AllTrackKeys[k] == track_attribute:

                    for attr, trackid in tqdm(
                        zip(AllTrackValues[k], AllTrackValues[p]),
                        total=len(AllTrackValues[k]),
                    ):

                        if math.isnan(trackid):
                            continue
                        else:
                            idattr[trackid] = attr

            for count, k in enumerate(range(len(AllKeys))):
                yield count
                locations = []
                if AllKeys[k] == "TRACK_ID":

                    for trackid, time, z, y, x in tqdm(
                        zip(
                            AllValues[k],
                            AllValues[keyT],
                            AllValues[keyZ],
                            AllValues[keyY],
                            AllValues[keyX],
                        ),
                        total=len(AllValues[k]),
                    ):

                        if len(plugin.seg_image.value.shape) == 4:
                            centroid = (time, z, y, x)
                        else:
                            centroid = (time, y, x)

                        attr = idattr[trackid]
                        locations.append([attr, centroid])

        new_seg_image = Relabel(plugin.seg_image.value.copy(), locations)

        return new_seg_image, attribute

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
            visible=True,
            label="TrackMate xml",
            mode="r",
        ),
        track_csv=dict(
            widget_type="FileEdit", visible=True, label="Track csv", mode="r"
        ),
        spot_csv=dict(
            widget_type="FileEdit", visible=True, label="Spot csv", mode="r"
        ),
        edges_csv=dict(
            widget_type="FileEdit",
            visible=True,
            label="Edges/Links csv",
            mode="r",
        ),
        axes=dict(
            widget_type="LineEdit",
            label="Image Axes",
            value=DEFAULTS_MODEL["axes"],
        ),
        defaults_model_button=dict(
            widget_type="PushButton", text="Restore Model Defaults"
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

        spot_dataset, spot_dataset_index = get_csv_data(spot_csv)

        track_dataset, track_dataset_index = get_csv_data(track_csv)

        edges_dataset, edges_dataset_index = get_csv_data(edges_csv)

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
    _color_tracks_tab_layout.addWidget(plugin_color_parameters.native)
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

    @change_handler(
        plugin_color_parameters.spot_attributes,
        plugin_color_parameters.track_attributes,
    )
    def _spot_track_attribute_color():

        if (
            plugin.color_parameters.spot_attributes.value
            is not AttributeBoxname
        ):
            plugin_color_parameters.track_attributes.value = (
                TrackAttributeBoxname
            )
        if (
            plugin.color_parameters.track_attributes.value
            is not TrackAttributeBoxname
        ):
            plugin_color_parameters.spot_attributes.value = AttributeBoxname

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
