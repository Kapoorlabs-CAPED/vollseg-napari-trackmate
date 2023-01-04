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

    from csbdeep.utils import axes_dict
    from napari.qt.threading import thread_worker
    from napatrackmater.bTrackmate import normalizeZeroOne
    from skimage.util import map_array

    from ._data_model import pandasModel
    from ._table_widget import TrackTable

    DEBUG = False
    scale = 255 * 255
    # Boxname = "TrackBox"
    AttributeBoxname = "AttributeIDBox"
    TrackAttributeBoxname = "TrackAttributeIDBox"

    track_analysis_spot_keys = dict(
        spot_id="ID",
        track_id="TRACK_ID",
        quality="QUALITY",
        posix="POSITION_X",
        posiy="POSITION_Y",
        posiz="POSITION_Z",
        posit="POSITION_T",
        frame="FRAME",
        radius="RADIUS",
        mean_intensity_ch1="MEAN_INTENSITY_CH1",
        total_intensity_ch1="TOTAL_INTENSITY_CH1",
        mean_intensity_ch2="MEAN_INTENSITY_CH2",
        total_intensity_ch2="TOTAL_INTENSITY_CH2",
    )
    track_analysis_edges_keys = dict(
        spot_source_id="SPOT_SOURCE_ID",
        spot_target_id="SPOT_TARGET_ID",
        directional_change_rate="DIRECTIONAL_CHANGE_RATE",
        speed="SPEED",
        displacement="DISPLACEMENT",
        edge_time="EDGE_TIME",
        edge_x_location="EDGE_X_LOCATION",
        edge_y_location="EDGE_Y_LOCATION",
        edge_z_location="EDGE_Z_LOCATION",
    )
    track_analysis_track_keys = dict(
        number_spots="NUMBER_SPOTS",
        number_gaps="NUMBER_GAPS",
        number_splits="NUMBER_SPLITS",
        number_merges="NUMBER_MERGES",
        track_duration="TRACK_DURATION",
        track_start="TRACK_START",
        track_stop="TRACK_STOP",
        track_displacement="TRACK_DISPLACEMENT",
        track_x_location="TRACK_X_LOCATION",
        track_y_location="TRACK_Y_LOCATION",
        track_z_location="TRACK_Z_LOCATION",
        track_mean_speed="TRACK_MEAN_SPEED",
        track_max_speed="TRACK_MAX_SPEED",
        track_min_speed="TRACK_MIN_SPEED",
        track_median_speed="TRACK_MEDIAN_SPEED",
        track_std_speed="TRACK_STD_SPEED",
        track_mean_quality="TRACK_MEAN_QUALITY",
        total_track_distance="TOTAL_DISTANCE_TRAVELED",
        max_track_distance="MAX_DISTANCE_TRAVELED",
        mean_straight_line_speed="MEAN_STRAIGHT_LINE_SPEED",
        linearity_forward_progression="LINEARITY_OF_FORWARD_PROGRESSION",
        mean_directional_change_rate="MEAN_DIRECTIONAL_CHANGE_RATE",
    )

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

        nonlocal xcalibration, ycalibration, zcalibration, tcalibration, filtered_track_ids, tracks, settings

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

    def get_label_data(image, debug=DEBUG):

        image = image.data[0] if image.multiscale else image.data
        if debug:
            print("Label image loaded")
        return np.asarray(image).astype(np.uint16)

    def get_csv_data(csv):

        dataset = pd.read_csv(
            csv, delimiter=",", encoding="unicode_escape", low_memory=False
        )[3:]
        dataset_index = dataset.index
        print(dataset.head())
        return dataset, dataset_index

    def get_track_dataset(track_dataset, track_dataset_index):

        nonlocal AllTrackValues
        AllTrackValues = {}
        track_id = track_analysis_spot_keys["track_id"]
        Tid = track_dataset[track_id].astype("float")
        indices = np.where(Tid == 0)
        maxtrack_id = max(Tid)
        condition_indices = track_dataset_index[indices]
        Tid[condition_indices] = maxtrack_id + 1
        AllTrackValues[track_id] = Tid
        for k in track_analysis_track_keys.values():

            if k != track_id:
                x = track_dataset[k].astype("float")
                minval = min(x)
                maxval = max(x)

                if minval > 0 and maxval <= 1:

                    x = normalizeZeroOne(x, scale=scale)

                AllTrackValues[k] = x

        TrackAttributeids = []
        TrackAttributeids.append(TrackAttributeBoxname)
        for attributename in track_analysis_track_keys.keys():
            TrackAttributeids.append(attributename)

        plugin_color_parameters.track_attributes.choices = TrackAttributeids

    def get_edges_dataset(edges_dataset, edges_dataset_index):

        nonlocal AllEdgesValues
        AllEdgesValues = {}
        track_id = track_analysis_spot_keys["track_id"]
        Tid = edges_dataset[track_id].astype("float")
        indices = np.where(Tid == 0)
        maxtrack_id = max(Tid)
        condition_indices = edges_dataset_index[indices]
        Tid[condition_indices] = maxtrack_id + 1
        AllEdgesValues[track_id] = Tid

        for k in track_analysis_edges_keys.values():

            if k != track_id:
                x = edges_dataset[k].astype("float")

                AllEdgesValues[k] = x

    def get_spot_dataset(spot_dataset, spot_dataset_index):

        nonlocal AllValues
        AllValues = {}
        track_id = track_analysis_spot_keys["track_id"]
        posix = track_analysis_spot_keys["posix"]
        posiy = track_analysis_spot_keys["posiy"]
        posiz = track_analysis_spot_keys["posiz"]
        frame = track_analysis_spot_keys["frame"]
        Tid = spot_dataset[track_id].astype("float")
        indices = np.where(Tid == 0)
        maxtrack_id = max(Tid)
        condition_indices = spot_dataset_index[indices]
        Tid[condition_indices] = maxtrack_id + 1

        AllValues[track_id] = Tid
        LocationX = (
            spot_dataset[posix].astype("float") / xcalibration
        ).astype("int")
        LocationY = (
            spot_dataset[posiy].astype("float") / ycalibration
        ).astype("int")
        LocationZ = (
            spot_dataset[posiz].astype("float") / zcalibration
        ).astype("int")
        LocationT = (spot_dataset[frame].astype("float")).astype("int")
        AllValues[posix] = LocationX
        AllValues[posiy] = LocationY
        AllValues[posiz] = LocationZ
        AllValues[frame] = LocationT

        for k in track_analysis_spot_keys.values():

            if (
                k != track_id
                and k != posix
                and k != posiy
                and k != posiz
                and k != frame
            ):

                AllValues[k] = spot_dataset[k].astype("float")

        Attributeids = []
        Attributeids.append(AttributeBoxname)
        for attributename in track_analysis_spot_keys.keys():
            Attributeids.append(attributename)
        plugin_color_parameters.spot_attributes.choices = Attributeids

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

    AllTrackValues = {}
    # AllTrackID = []
    # AllTrackAttr = []
    AllValues = {}
    AllEdgesValues = {}
    filtered_track_ids = []
    xcalibration = 1
    ycalibration = 1
    zcalibration = 1
    tcalibration = 1
    tracks = None
    settings = None
    track_model_type_choices = [
        ("Dividing", "Dividing"),
        ("Non-Dividing", "Non-Dividing"),
        ("Both", "Both"),
    ]

    DEFAULTS_MODEL = dict(axes="TZYX", track_model_type="Both")
    model_selected_track = DEFAULTS_MODEL["track_model_type"]
    DEFAULTS_FUNC_PARAMETERS = dict()

    def get_model_track(track_model_type):

        return track_model_type

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
            choices=[AttributeBoxname],
            value=AttributeBoxname,
            label="Spot Attributes",
        ),
        track_attributes=dict(
            widget_type="ComboBox",
            visible=True,
            choices=[TrackAttributeBoxname],
            value=TrackAttributeBoxname,
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

        if plugin.track_csv.value is not None:

            _load_track_csv(plugin.track_csv.value)

        if plugin.spot_csv.value is not None:

            _load_spot_csv(plugin.spot_csv.value)

        worker = _Color_tracks(spot_attributes, track_attributes)
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

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.25080/majora-1b6fd038-014")

    def return_color_tracks(new_seg_image, attribute):

        plugin.viewer.value.add_labels(new_seg_image, name=attribute)

    @thread_worker(connect={"returned": return_color_tracks})
    def _Color_tracks(spot_attribute, track_attribute):

        yield 0
        x_seg = get_label_data(plugin.seg_image.value)
        posix = track_analysis_spot_keys["posix"]
        posiy = track_analysis_spot_keys["posiy"]
        posiz = track_analysis_spot_keys["posiz"]
        frame = track_analysis_spot_keys["frame"]
        track_id = track_analysis_spot_keys["track_id"]
        if spot_attribute is not None:

            attribute = spot_attribute

            for count, k in enumerate(track_analysis_spot_keys.keys()):
                yield count
                locations = []
                if track_analysis_spot_keys[k] == spot_attribute:

                    for attr, time, z, y, x in tqdm(
                        zip(
                            AllValues[k],
                            AllValues[frame],
                            AllValues[posiz],
                            AllValues[posiy],
                            AllValues[posix],
                        ),
                        total=len(AllValues[k]),
                    ):
                        if len(plugin.seg_image.value.shape) == 4:
                            centroid = (time, z, y, x)
                        else:
                            centroid = (time, y, x)

                        locations.append([attr, centroid])

        if track_attribute is not None:

            attribute = track_attribute
            idattr = {}

            for k in track_analysis_track_keys.keys():

                if track_analysis_track_keys[k] == track_attribute:

                    for attr, trackid in tqdm(
                        zip(AllTrackValues[k], AllTrackValues[track_id]),
                        total=len(AllTrackValues[k]),
                    ):

                        if math.isnan(trackid):
                            continue
                        else:
                            idattr[trackid] = attr

            locations = []
            for trackid, time, z, y, x in tqdm(
                zip(
                    AllValues[track_id],
                    AllValues[frame],
                    AllValues[posiz],
                    AllValues[posiy],
                    AllValues[posix],
                ),
                total=len(AllValues[track_id]),
            ):

                if len(x_seg.shape) == 4:
                    centroid = (time, z, y, x)
                else:
                    centroid = (time, y, x)

                attr = idattr[trackid]
                locations.append([attr, centroid])

        new_seg_image = Relabel(plugin.x_seg.copy(), locations)

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
        track_model_type=dict(
            widget_type="RadioButtons",
            label="Track Model Type",
            orientation="horizontal",
            choices=track_model_type_choices,
            value=DEFAULTS_MODEL["track_model_type"],
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
        track_model_type,
        defaults_model_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        if image is not None:
            x = get_data(image)
            print(x.shape)
        if seg_image is not None:
            x_seg = get_label_data(seg_image)
            print(x_seg.shape)
        if mask_image is not None:
            x_mask = get_label_data(mask_image)
            print(x_mask.shape)

        nonlocal worker

        track_model = get_model_track(model_selected_track)
        print(track_model)
        worker = _refreshStatPlotData(xml_path, spot_csv, track_csv, edges_csv)
        worker.start()

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

    stat_canvas = FigureCanvas()
    stat_canvas.figure.set_tight_layout(True)
    stat_ax = stat_canvas.figure.subplots(3, 3)

    stat_plot_tab = stat_canvas
    _stat_plot_tab_layout = QVBoxLayout()
    stat_plot_tab.setLayout(_stat_plot_tab_layout)
    _stat_plot_tab_layout.addWidget(stat_plot_tab)
    tabs.addTab(stat_plot_tab, "Temporal Statistics")

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
            # myTableData = getLayerDataFrame(rowList=addedRowList)
            myTableData = df
            table_tab.myModel.myAppendRow(myTableData)
            _selectInTable(selection)
            table_tab.signalDataChanged.emit(action, selection, df)
        elif action == "delete":
            # was this
            deleteRowSet = selection
            # logger.info(f'myEventType:{myEventType} deleteRowSet:{deleteRowSet}')
            # deletedDataFrame = myTable2.myModel.myGetData().iloc[list(deleteRowSet)]

            _deleteRows(deleteRowSet)

            # _blockDeleteFromTable = True
            # myTable2.myModel.myDeleteRows(deleteRowList)
            # _blockDeleteFromTable = False

            table_tab.signalDataChanged.emit(action, selection, df)
        elif action == "change":
            moveRowList = list(selection)  # rowList is actually indexes
            myTableData = df
            # myTableData = getLayerDataFrame(rowList=moveRowList)
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

    @thread_worker()
    def _refreshStatPlotData(xml_path, spot_csv, track_csv, edges_csv):

        get_xml_data(xml_path)
        spot_dataset, spot_dataset_index = get_csv_data(spot_csv)
        track_dataset, track_dataset_index = get_csv_data(track_csv)
        edges_dataset, edges_dataset_index = get_csv_data(edges_csv)
        get_track_dataset(track_dataset, track_dataset_index)
        get_spot_dataset(spot_dataset, spot_dataset_index)
        get_edges_dataset(edges_dataset, edges_dataset_index)

        Attr = {}

        frameid_key = track_analysis_spot_keys["frame"]
        zposid_key = track_analysis_spot_keys["posiz"]
        yposid_key = track_analysis_spot_keys["posiy"]
        xposid_key = track_analysis_spot_keys["posix"]
        spotid_key = track_analysis_spot_keys["spot_id"]
        trackid_key = track_analysis_spot_keys["track_id"]
        radius_key = track_analysis_spot_keys["radius"]
        mean_intensity_ch1 = track_analysis_spot_keys["mean_intensity_ch1"]
        mean_intensity_ch2 = track_analysis_spot_keys["mean_intensity_ch2"]

        sourceid_key = track_analysis_edges_keys["spot_source_id"]
        dcr_key = track_analysis_edges_keys["directional_change_rate"]
        speed_key = track_analysis_edges_keys["speed"]
        disp_key = track_analysis_edges_keys["displacement"]

        starttime = int(min(AllValues[frameid_key]))
        endtime = int(max(AllValues[frameid_key]))
        print(starttime, endtime, "ss")

        for (
            sourceid,
            dcrid,
            speedid,
            dispid,
            zposid,
            yposid,
            xposid,
            radiusid,
            meanintensitych1id,
            meanintensitych2id,
        ) in zip(
            AllEdgesValues[sourceid_key],
            AllEdgesValues[dcr_key],
            AllEdgesValues[speed_key],
            AllEdgesValues[disp_key],
            AllValues[zposid_key],
            AllValues[yposid_key],
            AllValues[xposid_key],
            AllValues[radius_key],
            AllValues[mean_intensity_ch1],
            AllValues[mean_intensity_ch2],
        ):

            Attr[int(sourceid)] = [
                dcrid,
                speedid,
                dispid,
                zposid,
                yposid,
                xposid,
                radiusid,
                meanintensitych1id,
                meanintensitych2id,
            ]

        Timedcr = []
        Timespeed = []
        Timeradius = []
        TimeCurmeaninch2 = []
        TimeCurmeaninch1 = []
        Timedisppos = []
        Timedispneg = []

        Timedispposy = []
        Timedispnegy = []

        Timedispposx = []
        Timedispnegx = []

        Alldcrmean = []
        Allspeedmean = []
        Allradiusmean = []
        AllCurmeaninch1mean = []
        AllCurmeaninch2mean = []
        Alldispmeanpos = []
        Alldispmeanneg = []

        Alldispmeanposx = []
        Alldispmeanposy = []

        Alldispmeannegx = []
        Alldispmeannegy = []

        Alldcrvar = []
        Allspeedvar = []
        Allradiusvar = []
        AllCurmeaninch1var = []
        AllCurmeaninch2var = []
        Alldispvarpos = []
        Alldispvarneg = []

        Alldispvarposy = []
        Alldispvarnegy = []

        Alldispvarposx = []
        Alldispvarnegx = []

        for i in tqdm(range(starttime, endtime), total=endtime - starttime):

            Curdcr = []
            Curspeed = []
            Curdisp = []
            Curdispz = []
            Curdispy = []
            Curdispx = []
            Currpos = []
            Curmeaninch1 = []
            Curmeaninch2 = []
            for spotid, trackid, frameid in zip(
                AllValues[spotid_key],
                AllValues[trackid_key],
                AllValues[frameid_key],
            ):

                if i == int(frameid):
                    if int(spotid) in Attr:
                        (
                            dcr,
                            speed,
                            disp,
                            zpos,
                            ypos,
                            xpos,
                            rpos,
                            meaninch1pos,
                            meaninch2pos,
                        ) = Attr[int(spotid)]
                        if dcr is not None:
                            Curdcr.append(dcr)

                        if speed is not None:
                            Curspeed.append(speed)
                        if disp is not None:
                            Curdisp.append(disp)
                        if zpos is not None:
                            Curdispz.append(zpos)
                        if ypos is not None:
                            Curdispy.append(ypos)

                        if xpos is not None:
                            Curdispx.append(xpos)
                        if rpos is not None:
                            Currpos.append(rpos)
                        if meaninch1pos is not None:
                            Curmeaninch1.append(meaninch1pos)

                        if meaninch2pos is not None:
                            Curmeaninch2.append(meaninch2pos)

            dispZ = np.diff(Curdispz)
            dispY = np.diff(Curdispy)
            dispX = np.diff(Curdispx)

            meanCurdcr = np.mean(Curdcr)
            varCurdcr = np.var(Curdcr)
            if meanCurdcr is not None:
                Alldcrmean.append(meanCurdcr)
                Alldcrvar.append(varCurdcr)
                Timedcr.append(i * tcalibration)

            meanCurspeed = np.mean(Curspeed)
            varCurspeed = np.var(Curspeed)
            if meanCurspeed is not None:

                Allspeedmean.append(meanCurspeed)
                Allspeedvar.append(varCurspeed)
                Timespeed.append(i * tcalibration)

            meanCurrpos = np.mean(Currpos)
            varCurrpos = np.var(Currpos)
            if meanCurrpos is not None:

                Allradiusmean.append(meanCurrpos)
                Allradiusvar.append(varCurrpos)
                Timeradius.append(i * tcalibration)

            meanCurmeaninch1 = np.mean(Curmeaninch1)
            varCurmeaninch1 = np.var(Curmeaninch1)
            if meanCurmeaninch1 is not None:

                AllCurmeaninch1mean.append(meanCurmeaninch1)
                AllCurmeaninch1var.append(varCurmeaninch1)
                TimeCurmeaninch1.append(i * tcalibration)

            meanCurmeaninch2 = np.mean(Curmeaninch2)
            varCurmeaninch2 = np.var(Curmeaninch2)
            if meanCurmeaninch2 is not None:

                AllCurmeaninch2mean.append(meanCurmeaninch2)
                AllCurmeaninch2var.append(varCurmeaninch2)
                TimeCurmeaninch2.append(i * tcalibration)

            meanCurdisp = np.mean(dispZ)
            varCurdisp = np.var(dispZ)

            meanCurdispy = np.mean(dispY)
            varCurdispy = np.var(dispY)

            meanCurdispx = np.mean(dispX)
            varCurdispx = np.var(dispX)

            if meanCurdisp is not None:
                if meanCurdisp >= 0:
                    Alldispmeanpos.append(meanCurdisp)
                    Alldispvarpos.append(varCurdisp)
                    Timedisppos.append(i * tcalibration)
                elif meanCurdisp < 0:
                    Alldispmeanneg.append(meanCurdisp)
                    Alldispvarneg.append(varCurdisp)
                    Timedispneg.append(i * tcalibration)

            if meanCurdispy is not None:
                if meanCurdispy >= 0:
                    Alldispmeanposy.append(meanCurdispy)
                    Alldispvarposy.append(varCurdispy)
                    Timedispposy.append(i * tcalibration)
                elif meanCurdispy < 0:
                    Alldispmeannegy.append(meanCurdispy)
                    Alldispvarnegy.append(varCurdispy)
                    Timedispnegy.append(i * tcalibration)

            if meanCurdispx is not None:
                if meanCurdispx >= 0:
                    Alldispmeanposx.append(meanCurdispx)
                    Alldispvarposx.append(varCurdispx)
                    Timedispposx.append(i * tcalibration)
                elif meanCurdispx < 0:
                    Alldispmeannegx.append(meanCurdispx)
                    Alldispvarnegx.append(varCurdispx)
                    Timedispnegx.append(i * tcalibration)
        for i in range(stat_ax.shape[0]):
            for j in range(stat_ax.shape[1]):
                stat_ax[i, j].cla()

        stat_ax[0, 0].errorbar(
            Timespeed,
            Allspeedmean,
            Allspeedvar,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )
        stat_ax[0, 0].set_title("Speed")
        stat_ax[0, 0].set_xlabel("Time (min)")
        stat_ax[0, 0].set_ylabel("um/min")

        stat_ax[0, 1].errorbar(
            Timeradius,
            Allradiusmean,
            Allradiusvar,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )
        stat_ax[0, 1].set_title("Radius")
        stat_ax[0, 1].set_xlabel("Time (min)")
        stat_ax[0, 1].set_ylabel("um")

        stat_ax[1, 0].errorbar(
            Timedisppos,
            Alldispmeanpos,
            Alldispvarpos,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )
        stat_ax[1, 0].errorbar(
            Timedispneg,
            Alldispmeanneg,
            Alldispvarneg,
            linestyle="None",
            marker=".",
            mfc="red",
            ecolor="red",
        )
        stat_ax[1, 0].set_title("Displacement in Z")
        stat_ax[1, 0].set_xlabel("Time (min)")
        stat_ax[1, 0].set_ylabel("um")

        stat_ax[1, 1].errorbar(
            Timedispposy,
            Alldispmeanposy,
            Alldispvarposy,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )
        stat_ax[1, 1].errorbar(
            Timedispnegy,
            Alldispmeannegy,
            Alldispvarnegy,
            linestyle="None",
            marker=".",
            mfc="red",
            ecolor="red",
        )
        stat_ax[1, 1].set_title("Displacement in Y")
        stat_ax[1, 1].set_xlabel("Time (min)")
        stat_ax[1, 1].set_ylabel("um")

        stat_ax[1, 2].errorbar(
            Timedispposx,
            Alldispmeanposx,
            Alldispvarposx,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )
        stat_ax[1, 2].errorbar(
            Timedispnegx,
            Alldispmeannegx,
            Alldispvarnegx,
            linestyle="None",
            marker=".",
            mfc="red",
            ecolor="red",
        )
        stat_ax[1, 2].set_title("Displacement in X")
        stat_ax[1, 2].set_xlabel("Time (min)")
        stat_ax[1, 2].set_ylabel("um")

        stat_ax[2, 1].errorbar(
            TimeCurmeaninch1,
            AllCurmeaninch1mean,
            AllCurmeaninch1var,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )
        stat_ax[2, 1].set_title("Intensity ch1")
        stat_ax[2, 1].set_xlabel("Time (min)")
        stat_ax[2, 1].set_ylabel("")

        stat_ax[0, 2].errorbar(
            TimeCurmeaninch2,
            AllCurmeaninch2mean,
            AllCurmeaninch2var,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )
        stat_ax[0, 2].set_title("Intensity ch2")
        stat_ax[0, 2].set_xlabel("Time (min)")
        stat_ax[0, 2].set_ylabel("")

        stat_canvas.draw()

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

    def select_model_track(key):
        nonlocal model_selected_track
        model_selected_track = key

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

    @change_handler(plugin.track_csv, init=False)
    def _load_track_csv(path: str):

        track_dataset, track_dataset_index = get_csv_data(path)
        get_track_dataset(track_dataset, track_dataset_index)

    @change_handler(plugin.spot_csv, init=False)
    def _load_spot_csv(path: str):

        spot_dataset, spot_dataset_index = get_csv_data(path)
        get_spot_dataset(spot_dataset, spot_dataset_index)

    @change_handler(plugin.edges_csv, init=False)
    def _load_edges_csv(path: str):

        edges_dataset, edges_dataset_index = get_csv_data(path)
        get_edges_dataset(edges_dataset, edges_dataset_index)

    @change_handler(
        plugin_color_parameters.spot_attributes,
        init=False,
    )
    def _spot_attribute_color(value):

        plugin_color_parameters.spot_attributes.value = value

    @change_handler(
        plugin_color_parameters.track_attributes,
        init=False,
    )
    def _track_attribute_color(value):

        plugin_color_parameters.track_attributes.value = value

    @change_handler(plugin.track_model_type, init=False)
    def _track_model_type_change():

        key = plugin.track_model_type.value
        select_model_track(key)

    @change_handler(
        plugin_function_parameters.defaults_params_button, init=False
    )
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

        ndim = get_data(image).ndim
        if ndim == 4:
            axes = "TZYX"
        if ndim == 3:
            axes = "TYX"
        if ndim == 2:
            axes = "YX"
        else:
            axes = "TZYX"
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
