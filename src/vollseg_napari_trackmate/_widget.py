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
from caped_ai_tabulour._tabulour import Tabulour
from magicgui import magicgui
from magicgui import widgets as mw
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget
from tqdm import tqdm


def plugin_wrapper_track():

    pass

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
        originallabels = []
        newlabels = []
        for relabelval, centroid in locations:

            if len(Newseg_image.shape) == 4:
                time, z, y, x = centroid
            else:
                time, y, x = centroid

            if len(Newseg_image.shape) == 4:
                originallabel = Newseg_image[time, z, y, x]
            else:
                originallabel = Newseg_image[time, y, x]

            if originallabel == 0:
                relabelval = 0
            if math.isnan(relabelval):
                relabelval = -1
            originallabels.append(int(originallabel))
            newlabels.append(int(relabelval))

        relabeled = map_array(
            Newseg_image, np.asarray(originallabels), np.asarray(newlabels)
        )

        return relabeled

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

    def _refreshTrackData(unique_tracks, unique_tracks_properties):

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
        }
        for layer in list(plugin.viewer.value.layers):
            if "Track" == layer.name or "Track_points" == layer.name:
                plugin.viewer.value.layers.remove(layer)
        vertices = unique_tracks[:, 1:]
        plugin.viewer.value.add_points(vertices, size=2, name="Track_points")
        plugin.viewer.value.add_tracks(
            unique_tracks,
            name="Track",
            features=features,
        )

    def return_color_tracks(pred):

        if not isinstance(pred, int):
            new_seg_image, attribute = pred
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
            for count, k in enumerate(
                _trackmate_objects.track_analysis_spot_keys.keys()
            ):
                yield count
                locations = []
                if k == spot_attribute:

                    for attr, time, z, y, x in tqdm(
                        zip(
                            _trackmate_objects.AllValues[k],
                            _trackmate_objects.AllValues[frame],
                            _trackmate_objects.AllValues[posiz],
                            _trackmate_objects.AllValues[posiy],
                            _trackmate_objects.AllValues[posix],
                        ),
                        total=len(_trackmate_objects.AllValues[k]),
                    ):
                        if len(x_seg.shape) == 4:
                            centroid = (time, z, y, x)
                        else:
                            centroid = (time, y, x)

                        locations.append([attr, centroid])

        if track_attribute != TrackAttributeBoxname:

            attribute = track_attribute
            idattr = {}

            for k in _trackmate_objects.track_analysis_track_keys.keys():

                if k == track_attribute:

                    for attr, trackid in tqdm(
                        zip(
                            _trackmate_objects.AllTrackValues[k],
                            _trackmate_objects.AllTrackValues[track_id],
                        ),
                        total=len(_trackmate_objects.AllTrackValues[k]),
                    ):
                        if math.isnan(trackid):
                            continue
                        else:
                            idattr[trackid] = attr

            locations = []
            for trackid, time, z, y, x in tqdm(
                zip(
                    _trackmate_objects.AllValues[track_id],
                    _trackmate_objects.AllValues[frame],
                    _trackmate_objects.AllValues[posiz],
                    _trackmate_objects.AllValues[posiy],
                    _trackmate_objects.AllValues[posix],
                ),
                total=len(_trackmate_objects.AllValues[track_id]),
            ):

                if len(x_seg.shape) == 4:
                    centroid = (time, z, y, x)
                else:
                    centroid = (time, y, x)

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
            choices=[TrackidBox],
            value=TrackidBox,
        ),
        defaults_model_button=dict(
            widget_type="PushButton", text="Restore Model Defaults"
        ),
        show_track_id_button=dict(
            widget_type="PushButton", text="Display Selected Tracks"
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
        show_track_id_button,
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

        def progress_thread(current_time):

            progress_bar.label = "Analyzing Tracks"
            # progress_bar.range = (0, n_frames - 1)
            progress_bar.value = current_time
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
        worker = _refreshStatPlotData()
        worker.returned.connect(plot_main)

        select_track_nature()

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

    plugin.show_track_id_button.native.setStyleSheet(
        "background-color: orange"
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

    hist_plot_class = TemporalStatistics(tabs)
    hist_plot_tab = hist_plot_class.stat_plot_tab
    tabs.addTab(hist_plot_tab, "Histogram Statistics")

    stat_plot_class = TemporalStatistics(tabs)
    stat_plot_tab = stat_plot_class.stat_plot_tab
    tabs.addTab(stat_plot_tab, "Temporal Statistics")

    table_tab = Tabulour()
    tabs.addTab(table_tab, "Table")

    plugin.native.layout().addWidget(tabs)

    def plot_main():
        _refreshPlotData()

    def _refreshPlotData():

        trackid_key = _trackmate_objects.track_analysis_spot_keys["track_id"]
        for k in _trackmate_objects.AllTrackValues.keys():
            if k is not trackid_key:
                TrackAttr = []
                for attr, trackid in tqdm(
                    zip(
                        _trackmate_objects.AllTrackValues[k],
                        _trackmate_objects.AllTrackValues[trackid_key],
                    ),
                    total=len(_trackmate_objects.AllTrackValues[k]),
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

    @thread_worker(connect={"returned": [plot_main]})
    def _refreshStatPlotData():
        nonlocal _trackmate_objects
        hist_plot_class._reset_container(hist_plot_class.scroll_layout)
        stat_plot_class._reset_container(stat_plot_class.scroll_layout)

    def _refreshTableData(
        df: pd.DataFrame, unique_cells: dict, time_key: str, other_key: str
    ):
        """Refresh all data in table by setting its data model from provided dataframe.
        Args:
            df (pd.DataFrame): Pandas dataframe to refresh with.
        """

        if table_tab is None:
            return

        if df is None:
            return

        table_tab.data = df
        table_tab.viewer = plugin.viewer.value
        table_tab._unique_cells = unique_cells
        for layer in list(plugin.viewer.value.layers):
            if isinstance(layer, napari.layers.Tracks):
                table_tab.layer = layer

        table_tab.time_key = time_key
        table_tab.other_key = other_key
        table_tab._set_model()
        # current_cell_id = table_tab._unique_cell_val

    def select_track_nature():
        key = plugin.track_model_type.value
        nonlocal _trackmate_objects, _track_ids_analyze
        if _trackmate_objects is not None:
            if key == "Dividing":
                plugin.track_id_box.choices = TrackidBox
                plugin.track_id_box.choices = (
                    _trackmate_objects.DividingTrackIds
                )
                _track_ids_analyze = _trackmate_objects.DividingTrackIds.copy()
                if "" in _track_ids_analyze:
                    _track_ids_analyze.remove("")
                if TrackidBox in _track_ids_analyze:
                    _track_ids_analyze.remove(TrackidBox)
            if key == "Non-Dividing":
                plugin.track_id_box.choices = TrackidBox
                plugin.track_id_box.choices = _trackmate_objects.NormalTrackIds
                _track_ids_analyze = _trackmate_objects.NormalTrackIds.copy()
                if "" in _track_ids_analyze:
                    _track_ids_analyze.remove("")
                if TrackidBox in _track_ids_analyze:
                    _track_ids_analyze.remove(TrackidBox)
            if key == "Both":
                plugin.track_id_box.choices = TrackidBox
                plugin.track_id_box.choices = _trackmate_objects.AllTrackIds
                _track_ids_analyze = _trackmate_objects.AllTrackIds.copy()
                if TrackidBox in _track_ids_analyze:
                    _track_ids_analyze.remove(TrackidBox)
                if "" in _track_ids_analyze:
                    _track_ids_analyze.remove("")

            _track_ids_analyze = list(map(int, _track_ids_analyze))

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet(
                "" if valid else "background-color: red"
            )

    @change_handler(plugin.track_id_box, init=False)
    def _track_id_box_change(value):
        plugin.track_id_box.value = value

    @change_handler(
        plugin.show_track_id_button,
        init=False,
    )
    def _analyze_id():

        nonlocal _track_ids_analyze, _trackmate_objects
        if _trackmate_objects is not None and _track_ids_analyze is not None:

            track_id = plugin.track_id_box.value
            unique_tracks = []
            unique_tracks_properties = []

            if track_id not in TrackidBox and track_id not in "":
                _to_analyze = [int(track_id)]
            else:
                _to_analyze = _track_ids_analyze.copy()

            for unique_track_id in tqdm(_to_analyze):

                tracklets = _trackmate_objects.unique_tracks[unique_track_id]
                tracklets_properties = (
                    _trackmate_objects.unique_track_properties[unique_track_id]
                )
                unique_tracks.append(tracklets)
                unique_tracks_properties.append(tracklets_properties)

            unique_tracks = np.concatenate(unique_tracks, axis=0)
            print("concatenated tracks")
            unique_tracks_properties = np.concatenate(
                unique_tracks_properties, axis=0
            )
            print("concatenated properties")
            _refreshTrackData(unique_tracks, unique_tracks_properties)
        selected = plugin.track_model_type
        selected.changed(selected.value)
        if track_id is not None:
            plugin.track_id_box.value = track_id

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
