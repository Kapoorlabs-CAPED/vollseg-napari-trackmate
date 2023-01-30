"""
VollSeg Napari Track .

Made by Kapoorlabs, 2022
"""

import functools
import math
from pathlib import Path
from typing import List, Union

import napari
import numpy as np
import pandas as pd
import seaborn as sns
from caped_ai_tabulour._tabulour import Tabulour, pandasModel
from magicgui import magicgui
from magicgui import widgets as mw
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget


def plugin_wrapper_track():

    from csbdeep.utils import axes_dict
    from napari.qt.threading import thread_worker
    from napatrackmater.Trackmate import TrackMate
    from skimage.util import map_array

    from vollseg_napari_trackmate._temporal_plots import TemporalStatistics

    DEBUG = False
    # Boxname = "TrackBox"
    AttributeBoxname = "AttributeIDBox"
    TrackAttributeBoxname = "TrackAttributeIDBox"
    TrackidBox = "All"
    _dividing_choices = ()
    _current_choices = ()
    _normal_choices = ()
    _both_choices = ()
    _dividing_track_ids_analyze = ()
    _normal_track_ids_analyze = ()
    _both_track_ids_analyze = ()

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
        NewSegimage = image.copy()
        for p in range(0, NewSegimage.shape[0]):

            sliceimage = NewSegimage[p, :]
            originallabels = []
            newlabels = []
            for relabelval, centroid in locations:
                if len(NewSegimage.shape) == 4:
                    time, z, y, x = centroid
                else:
                    time, y, x = centroid

                if p == int(time):

                    if len(NewSegimage.shape) == 4:
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
            NewSegimage[p, :] = relabeled

        return NewSegimage

    def get_label_data(image, debug=DEBUG):

        image = image.data[0] if image.multiscale else image.data
        if debug:
            print("Label image loaded")
        return np.asarray(image).astype(np.uint16)

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
    _track_ids_analyze = None
    _trackmate_objects = None
    track_model_type_choices = [
        ("Dividing", "Dividing"),
        ("Non-Dividing", "Non-Dividing"),
        ("Both", "Both"),
    ]

    DEFAULTS_MODEL = dict(axes="TZYX", track_model_type="Both")
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

        worker = _Color_tracks(spot_attributes, track_attributes)
        worker.returned.connect(return_color_tracks)
        if "T" in plugin.axes.value:
            t = axes_dict(plugin.axes.value)["T"]
            if plugin.image.value is not None:
                n_frames = get_data(plugin.image.value).shape[t]
            else:
                n_frames = get_label_data(plugin.seg_image.value).shape[t]

            def progress_thread(current_time):

                progress_bar.label = "Coloring cells with chosen attribute"
                progress_bar.range = (0, n_frames - 1)
                progress_bar.value = current_time
                progress_bar.show()

            worker.yielded.connect(return_color_tracks)

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.25080/majora-1b6fd038-014")

    def _refreshTrackData(pred):
        unique_tracks, unique_tracks_properties = pred
        features = {
            "time": map(int, np.asarray(unique_tracks_properties)[:, 0]),
            "generation": map(int, np.asarray(unique_tracks_properties)[:, 1]),
            "speed": map(float, np.asarray(unique_tracks_properties)[:, 2]),
            "directional_change_rate": map(
                float, np.asarray(unique_tracks_properties)[:, 3]
            ),
            "mean-intensity_ch1": map(
                float, np.asarray(unique_tracks_properties)[:, 4]
            ),
            "mean-intensity_ch2": map(
                float, np.asarray(unique_tracks_properties)[:, 5]
            ),
            "radius_pixels": map(
                float,
                np.asarray(unique_tracks_properties)[:, 6],
            ),
        }
        for layer in list(plugin.viewer.value.layers):
            if (
                "Track" == layer.name
                or "Boxes" == layer.name
                or "Track_points" == layer.name
            ):
                plugin.viewer.value.layers.remove(layer)
        vertices = unique_tracks[:, 1:]
        _boxes = []
        _sizes = []
        ndim = unique_tracks.shape[1] - 1
        for i in range(unique_tracks.shape[0]):
            # TZYX
            current_tracklet_location = unique_tracks[i][1:]
            current_tracklet_properties = unique_tracks_properties[i][-2:-1]
            _boxes.append([location for location in current_tracklet_location])
            _sizes.append([volume for volume in current_tracklet_properties])
        plugin.viewer.value.add_points(vertices, size=2, name="Track_points")
        plugin.viewer.value.add_points(
            np.array(_boxes),
            size=np.array(_sizes),
            name="Boxes",
            face_color=[0] * 4,
            edge_color="green",
            ndim=ndim,
        )
        plugin.viewer.value.add_tracks(
            unique_tracks,
            name="Track",
            features=features,
        )

    def return_color_tracks(pred):

        if not isinstance(pred, int):
            new_seg_image, attribute = pred
            new_seg_image = new_seg_image.astype("uint16")
            for layer in list(plugin.viewer.value.layers):
                if attribute in layer.name:
                    plugin.viewer.value.layers.remove(layer)
            plugin.viewer.value.add_labels(new_seg_image, name=attribute)

    @thread_worker(connect={"returned": return_color_tracks})
    def _Color_tracks(spot_attribute, track_attribute):
        nonlocal _trackmate_objects
        yield 0
        x_seg = get_label_data(plugin.seg_image.value)
        posix = _trackmate_objects.track_analysis_spot_keys["posix"]
        posiy = _trackmate_objects.track_analysis_spot_keys["posiy"]
        posiz = _trackmate_objects.track_analysis_spot_keys["posiz"]
        frame = _trackmate_objects.track_analysis_spot_keys["frame"]
        track_id = _trackmate_objects.track_analysis_spot_keys["track_id"]
        if spot_attribute != AttributeBoxname:

            attribute = spot_attribute
            locations = []

            for (k, v) in _trackmate_objects.unique_spot_properties.items():
                current_spot = _trackmate_objects.unique_spot_properties[k]
                z = int(
                    float(current_spot[posiz])
                    / _trackmate_objects.zcalibration
                )
                y = int(
                    float(current_spot[posiy])
                    / _trackmate_objects.ycalibration
                )
                x = int(
                    float(current_spot[posix])
                    / _trackmate_objects.xcalibration
                )
                time = int(float(current_spot[frame]))

                if spot_attribute in current_spot.keys():
                    attr = int(float(current_spot[spot_attribute]))
                    if len(x_seg.shape) == 4:
                        centroid = (time, z, y, x)
                    else:
                        centroid = (time, y, x)
                    locations.append([attr, centroid])

            new_seg_image = Relabel(x_seg.copy(), locations)

            pred = new_seg_image, attribute

        if track_attribute != TrackAttributeBoxname:

            attribute = track_attribute
            idattr = {}

            for k in _trackmate_objects.track_analysis_track_keys.keys():

                if k == track_attribute:

                    for attr, trackid in zip(
                        _trackmate_objects.AllTrackValues[k],
                        _trackmate_objects.AllTrackValues[track_id],
                    ):
                        if math.isnan(trackid):
                            continue
                        else:
                            idattr[trackid] = attr

            locations = []
            for (k, v) in _trackmate_objects.unique_spot_properties.items():
                current_spot = _trackmate_objects.unique_spot_properties[k]
                if track_id in current_spot.keys():
                    z = int(
                        float(current_spot[posiz])
                        / _trackmate_objects.zcalibration
                    )
                    y = int(
                        float(current_spot[posiy])
                        / _trackmate_objects.ycalibration
                    )
                    x = int(
                        float(current_spot[posix])
                        / _trackmate_objects.xcalibration
                    )
                    time = int(float(current_spot[frame]))

                    if len(x_seg.shape) == 4:
                        centroid = (time, z, y, x)
                    else:
                        centroid = (time, y, x)
                    trackid = int(float(current_spot[track_id]))
                    attr = idattr[trackid]
                    locations.append([attr, centroid])

            new_seg_image = Relabel(x_seg.copy(), locations)

            pred = new_seg_image, attribute

        return pred

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
        track_csv_path=dict(
            widget_type="FileEdit", visible=True, label="Track csv", mode="r"
        ),
        spot_csv_path=dict(
            widget_type="FileEdit", visible=True, label="Spot csv", mode="r"
        ),
        edges_csv_path=dict(
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
        track_id_box=dict(
            widget_type="ComboBox",
            visible=True,
            label="Select Track ID to analyze",
            choices=_current_choices,
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
        image: Union[napari.layers.Image, None],
        seg_image: Union[napari.layers.Labels, None],
        mask_image: Union[napari.layers.Labels, None],
        xml_path,
        track_csv_path,
        spot_csv_path,
        edges_csv_path,
        axes,
        track_model_type,
        track_id_box,
        defaults_model_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        x = None
        x_seg = None
        x_mask = None
        if image is not None:
            x = get_data(image)
            print(x.shape)

        if seg_image is not None:
            x_seg = get_label_data(seg_image)
            print(x_seg.shape)
        if mask_image is not None:
            x_mask = get_label_data(mask_image)
            print(x_mask.shape)

        nonlocal worker, _trackmate_objects

        progress_bar.label = "Analyzing Tracks"

        # progress_bar.range = (0, n_frames - 1)
        progress_bar.value = 0
        progress_bar.show()

        _trackmate_objects = TrackMate(
            xml_path,
            spot_csv_path,
            track_csv_path,
            edges_csv_path,
            AttributeBoxname,
            TrackAttributeBoxname,
            TrackidBox,
            x,
            x_mask,
        )
        _refreshStatPlotData()

        plugin_color_parameters.track_attributes.choices = (
            _trackmate_objects.TrackAttributeids
        )
        plugin_color_parameters.spot_attributes.choices = (
            _trackmate_objects.Attributeids
        )

    plugin.label_head.value = '<br>Citation <tt><a href="https://doi.org/10.25080/majora-1b6fd038-014" style="color:gray;">NapaTrackMater Scipy</a></tt>'
    plugin.label_head.native.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
    )
    plugin.progress_bar.hide()

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

    hist_plot_class = TemporalStatistics(tabs)
    hist_plot_tab = hist_plot_class.stat_plot_tab
    tabs.addTab(hist_plot_tab, "Histogram Statistics")

    stat_plot_class = TemporalStatistics(tabs)
    stat_plot_tab = stat_plot_class.stat_plot_tab
    tabs.addTab(stat_plot_tab, "Temporal Statistics")

    table_tab = Tabulour()
    table_tab.clicked.connect(table_tab._on_user_click)
    table_tab.clicked.connect(table_tab._make_boxes)
    tabs.addTab(table_tab, "Table")

    plugin.native.layout().addWidget(tabs)

    def plot_main():

        trackid_key = _trackmate_objects.track_analysis_spot_keys["track_id"]
        for k in _trackmate_objects.AllTrackValues.keys():
            if k is not trackid_key:
                TrackAttr = []
                for attr, trackid in zip(
                    _trackmate_objects.AllTrackValues[k],
                    _trackmate_objects.AllTrackValues[trackid_key],
                ):

                    TrackAttr.append(float(attr))

                hist_plot_class._repeat_after_plot()
                hist_ax = hist_plot_class.stat_ax
                sns.histplot(TrackAttr, kde=True, ax=hist_ax)
                hist_ax.set_title(str(k))
        stat_plot_class._repeat_after_plot()
        stat_ax = stat_plot_class.stat_ax
        stat_ax.cla()

        stat_ax.errorbar(
            _trackmate_objects.Time,
            _trackmate_objects.Allspeedmean,
            _trackmate_objects.Allspeedvar,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )
        stat_ax.set_title("Speed")
        stat_ax.set_xlabel("Time (min)")
        stat_ax.set_ylabel("um/min")

        stat_plot_class._repeat_after_plot()
        stat_ax = stat_plot_class.stat_ax

        stat_ax.errorbar(
            _trackmate_objects.Time,
            _trackmate_objects.Allradiusmean,
            _trackmate_objects.Allradiusvar,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )
        stat_ax.set_title("Radius")
        stat_ax.set_xlabel("Time (min)")
        stat_ax.set_ylabel("um")

        stat_plot_class._repeat_after_plot()
        stat_ax = stat_plot_class.stat_ax

        stat_ax.errorbar(
            _trackmate_objects.Time,
            _trackmate_objects.Alldispmeanpos,
            _trackmate_objects.Alldispvarpos,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )

        stat_ax.set_title("Displacement in Z")
        stat_ax.set_xlabel("Time (min)")
        stat_ax.set_ylabel("um")

        stat_plot_class._repeat_after_plot()
        stat_ax = stat_plot_class.stat_ax

        stat_ax.errorbar(
            _trackmate_objects.Time,
            _trackmate_objects.Alldispmeanposy,
            _trackmate_objects.Alldispvarposy,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )

        stat_ax.set_title("Displacement in Y")
        stat_ax.set_xlabel("Time (min)")
        stat_ax.set_ylabel("um")

        stat_plot_class._repeat_after_plot()
        stat_ax = stat_plot_class.stat_ax

        stat_ax.errorbar(
            _trackmate_objects.Time,
            _trackmate_objects.Alldispmeanposx,
            _trackmate_objects.Alldispvarposx,
            linestyle="None",
            marker=".",
            mfc="green",
            ecolor="green",
        )

        stat_ax.set_title("Displacement in X")
        stat_ax.set_xlabel("Time (min)")
        stat_ax.set_ylabel("um")

        for layer in list(plugin.viewer.value.layers):
            if isinstance(layer, napari.layers.Tracks):
                table_tab.layer = layer

    def _refreshStatPlotData():
        nonlocal _trackmate_objects, _current_choices, _dividing_choices, _normal_choices, _both_choices, _dividing_track_ids_analyze, _normal_track_ids_analyze, _both_track_ids_analyze
        hist_plot_class._reset_container(hist_plot_class.scroll_layout)
        stat_plot_class._reset_container(stat_plot_class.scroll_layout)

        root_cells = []
        columns = None
        root_cells = None
        unique_cells = _trackmate_objects.unique_spot_properties
        unique_tracks = _trackmate_objects.unique_tracks
        unique_track_properties = _trackmate_objects.unique_track_properties
        time_key = _trackmate_objects.frameid_key
        id_key = _trackmate_objects.trackid_key
        size_key = _trackmate_objects.quality_key
        dividing_key = _trackmate_objects.dividing_key
        _dividing_choices = TrackidBox
        _dividing_choices = _trackmate_objects.DividingTrackIds

        _dividing_track_ids_analyze = (
            _trackmate_objects.DividingTrackIds.copy()
        )
        if None in _dividing_track_ids_analyze:
            _dividing_track_ids_analyze.remove(None)
        if TrackidBox in _dividing_track_ids_analyze:
            _dividing_track_ids_analyze.remove(TrackidBox)

        _normal_choices = TrackidBox
        _normal_choices = _trackmate_objects.NormalTrackIds
        _normal_track_ids_analyze = _trackmate_objects.NormalTrackIds.copy()
        if None in _normal_track_ids_analyze:
            _normal_track_ids_analyze.remove(None)
        if TrackidBox in _normal_track_ids_analyze:
            _normal_track_ids_analyze.remove(TrackidBox)

        _both_choices = TrackidBox
        _both_choices = _trackmate_objects.AllTrackIds
        _both_track_ids_analyze = _trackmate_objects.AllTrackIds.copy()
        if TrackidBox in _both_track_ids_analyze:
            _both_track_ids_analyze.remove(TrackidBox)
        if None in _both_track_ids_analyze:
            _both_track_ids_analyze.remove(None)
        plugin.progress_bar.range = (0, len(unique_cells) - 1)

        for count, (k, v) in enumerate(unique_cells.items()):

            if _trackmate_objects.beforeid_key in v.keys():
                is_root = v[_trackmate_objects.beforeid_key]

                if is_root is None:
                    if columns is None:
                        columns = [value for value in v.keys()]
                    float_list = list(v.values())
                    if root_cells is None:
                        root_cells = np.asarray(float_list)
                    else:
                        root_cells = np.vstack(
                            (root_cells, np.asarray(float_list))
                        )
            plugin.progress_bar.value = count
        print(f"Making pandas dataframe  {root_cells.shape}")
        columns[0] = "Root_Cell_ID"
        df = pd.DataFrame(root_cells, columns=columns, dtype=object)
        print("Making pandas Model")
        table_tab.data = pandasModel(df)
        table_tab.viewer = plugin.viewer.value
        table_tab.unique_tracks = unique_tracks
        table_tab.unique_track_properties = unique_track_properties
        table_tab.size_key = size_key
        table_tab.time_key = time_key
        table_tab.id_key = id_key
        table_tab.dividing_key = dividing_key
        table_tab.zcalibration = _trackmate_objects.zcalibration
        table_tab.ycalibration = _trackmate_objects.ycalibration
        table_tab.xcalibration = _trackmate_objects.xcalibration
        table_tab._plugin = plugin
        table_tab.normal_choices = _normal_choices
        table_tab.dividing_choices = _dividing_choices
        table_tab._set_model()

        select_track_nature()

        plot_main()

    def select_track_nature():
        key = plugin.track_model_type.value
        nonlocal _trackmate_objects, _track_ids_analyze, _dividing_track_ids_analyze, _normal_track_ids_analyze, _both_track_ids_analyze, _current_choices
        if _trackmate_objects is not None:
            if key == "Dividing":
                plugin.track_id_box.choices = _dividing_choices
                _track_ids_analyze = _dividing_track_ids_analyze
                _current_choices = _dividing_choices
            if key == "Non-Dividing":
                plugin.track_id_box.choices = _normal_choices
                _track_ids_analyze = _normal_track_ids_analyze
                _current_choices = _normal_choices
            if key == "Both":
                plugin.track_id_box.choices = _both_choices
                _track_ids_analyze = _both_track_ids_analyze
                _current_choices = _both_choices

            _track_ids_analyze = list(map(int, _track_ids_analyze))

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet(
                "" if valid else "background-color: red"
            )

    def show_track(track_id):

        unique_tracks = []
        unique_tracks_properties = []

        if track_id not in TrackidBox and track_id is not None:
            _to_analyze = [int(track_id)]
        else:
            _to_analyze = _track_ids_analyze.copy()

        for unique_track_id in _to_analyze:

            tracklets = _trackmate_objects.unique_tracks[unique_track_id]
            tracklets_properties = _trackmate_objects.unique_track_properties[
                unique_track_id
            ]
            unique_tracks.append(tracklets)
            unique_tracks_properties.append(tracklets_properties)

        unique_tracks = np.concatenate(unique_tracks, axis=0)
        unique_tracks_properties = np.concatenate(
            unique_tracks_properties, axis=0
        )
        pred = unique_tracks, unique_tracks_properties
        _refreshTrackData(pred)

    @change_handler(plugin.track_id_box, init=False)
    def _track_id_box_change():

        nonlocal worker
        value = plugin.track_id_box.value
        print(plugin.track_id_box.value)
        nonlocal _track_ids_analyze, _trackmate_objects
        if (
            _trackmate_objects is not None
            and _track_ids_analyze is not None
            and value is not None
        ):

            track_id = value
            show_track(track_id)

    @change_handler(plugin.track_model_type, init=False)
    def _change_track_model_type(value):

        plugin.track_model_type.value = value
        select_track_nature()

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

    @change_handler(
        plugin_function_parameters.defaults_params_button, init=False
    )
    def restore_function_parameters_defaults():
        for k, v in DEFAULTS_FUNC_PARAMETERS.items():
            getattr(plugin_function_parameters, k).value = v

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
