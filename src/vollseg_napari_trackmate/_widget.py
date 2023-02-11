"""
VollSeg Napari Track .
Made by Kapoorlabs, 2022
"""

import functools
import math
import os
from pathlib import Path
from typing import List, Union

import napari
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from caped_ai_tabulour._tabulour import Tabulour, pandasModel
from magicgui import magicgui
from magicgui import widgets as mw
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget


def plugin_wrapper_track():

    from cellshape_cloud import CloudAutoEncoder
    from cellshape_cluster import DeepEmbeddedClustering
    from napatrackmater.pretrained import get_registered_models
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

    (
        _models_cloud_auto_encoder,
        _aliases_cloud_auto_encoder,
    ) = get_registered_models(CloudAutoEncoder)

    _models_cluster, _aliases_cluster = get_registered_models(
        DeepEmbeddedClustering
    )

    models_cloud_auto_encoder = [
        (
            (
                _aliases_cloud_auto_encoder[m][0]
                if len(_aliases_cloud_auto_encoder[m]) > 0
                else m
            ),
            m,
        )
        for m in _models_cloud_auto_encoder
    ]

    models_cluster = [
        ((_aliases_cluster[m][0] if len(_aliases_cluster[m]) > 0 else m), m)
        for m in _models_cluster
    ]

    model_cloud_auto_encoder_configs = dict()

    model_selected_cloud_auto_encoder = None
    model_selected_cluster = None

    DEFAULTS_MODEL = dict(
        cloud_auto_encoder_model_type=CloudAutoEncoder,
        cluster_model_type=DeepEmbeddedClustering,
        model_cloud_auto_encoder=models_cloud_auto_encoder[0][0],
        model_cluster=models_cluster[0][0],
        model_cloud_auto_encoder_none="No(Encoder)",
        model_cluster_none="No(Cluster)",
        axes="TZYX",
        track_model_type="Both",
    )

    CUSTOM_MODEL_CLOUD_AUTO_ENCODER = "CUSTOM_MODEL_CLOUD_AUTO_ENCODER"
    CUSTOM_MODEL_CLUSTER = "CUSTOM_MODEL_CLUSTER"

    cloud_auto_encoder_model_type_choices = [
        ("PreTrained(Encoder)", CloudAutoEncoder),
        ("No(Encoder)", "No(Encoder)"),
        ("Custom Encoder", CUSTOM_MODEL_CLOUD_AUTO_ENCODER),
    ]
    cluster_model_type_choices = [
        ("PreTrained(Cluster)", DeepEmbeddedClustering),
        ("No(Cluster)", "No(Cluster)"),
        ("Custom Cluster", CUSTOM_MODEL_CLUSTER),
    ]

    @functools.lru_cache(maxsize=None)
    def get_model_cloud_auto_encoder(
        cloud_auto_encoder_model_type, model_cloud_auto_encoder
    ):
        if cloud_auto_encoder_model_type == CUSTOM_MODEL_CLOUD_AUTO_ENCODER:
            path_auto = Path(model_cloud_auto_encoder)
            path_auto.is_dir() or _raise(
                FileNotFoundError(f"{path_auto} is not a directory")
            )
            config_cloud_auto_encoder = model_cloud_auto_encoder_configs[
                (cloud_auto_encoder_model_type, model_cloud_auto_encoder)
            ]

            model_class_cloud_auto_encoder = CloudAutoEncoder
            autoencoder = model_class_cloud_auto_encoder(
                num_features=config_cloud_auto_encoder["num_features"],
                k=config_cloud_auto_encoder["k_nearest_neighbours"],
                encoder_type=config_cloud_auto_encoder["encoder_type"],
                decoder_type=config_cloud_auto_encoder["decoder_type"],
            )
            checkpoint = torch.load(
                os.path.join(path_auto.parent, path_auto.stem)
            )
            autoencoder.load_state_dict(checkpoint["model_state_dict"])
            return autoencoder

        elif (
            cloud_auto_encoder_model_type
            != DEFAULTS_MODEL["model_cloud_auto_encoder_none"]
        ):
            return cloud_auto_encoder_model_type.local_from_pretrained(
                model_cloud_auto_encoder
            )
        else:
            return None

    @functools.lru_cache(maxsize=None)
    def get_model_cluster(
        cloud_auto_encoder_model_type,
        model_cloud_auto_encoder,
        cluster_model_type,
        model_cluster,
    ):

        autoencoder = get_model_cloud_auto_encoder(
            cloud_auto_encoder_model_type, model_cloud_auto_encoder
        )
        if cluster_model_type == CUSTOM_MODEL_CLUSTER:
            path_cluster = Path(cluster_model_type)
            path_cluster.is_dir() or _raise(
                FileNotFoundError(f"{path_cluster} is not a directory")
            )

            checkpoint = torch.load(
                os.path.join(path_cluster.parent, path_cluster.stem)
            )
            num_clusters = checkpoint["model_state_dict"][
                "clustering_layer.weight"
            ].shape[0]

            model = DeepEmbeddedClustering(
                autoencoder=autoencoder, num_clusters=num_clusters
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model_class_cluster = DeepEmbeddedClustering
            return model_class_cluster

        elif cluster_model_type != DEFAULTS_MODEL["model_cluster_none"]:
            return cluster_model_type.local_from_pretrained(model_cluster)
        else:
            return None

    update_cloud_auto_encoder = None  # Updater_Auto_Encoder()

    def select_model_cloud_auto_encoder(key):
        nonlocal model_selected_cloud_auto_encoder
        if key is not None:
            model_selected_cloud_auto_encoder = key
            # config_cloud_auto_encoder = model_cloud_auto_encoder_configs.get(
            #    key
            # )
            update_cloud_auto_encoder("model_autoencoder")
        if (
            plugin.cloud_auto_encoder_model_type.value
            == DEFAULTS_MODEL["model_cloud_auto_encoder_none"]
        ):
            model_selected_cloud_auto_encoder = None

    _track_ids_analyze = None
    _to_analyze = None
    _trackmate_objects = None
    track_model_type_choices = [
        ("Dividing", "Dividing"),
        ("Non-Dividing", "Non-Dividing"),
        ("Both", "Both"),
    ]

    track_model_type_dict = {
        0: track_model_type_choices[0][0],
        1: track_model_type_choices[1][0],
        2: track_model_type_choices[2][0],
    }

    @magicgui(
        image=dict(label="Input Image"),
        seg_image=dict(label="Optional Segmentation Image"),
        channel_seg_image=dict(label="Second channel (new XML)"),
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
        layout="vertical",
        persist=True,
        call_button=True,
    )
    def plugin_data(
        image: Union[napari.layers.Image, None],
        seg_image: Union[napari.layers.Labels, None],
        channel_seg_image: Union[napari.layers.Labels, None],
        mask_image: Union[napari.layers.Labels, None],
        xml_path,
        track_csv_path,
        spot_csv_path,
        edges_csv_path,
        axes,
    ) -> List[napari.types.LayerDataTuple]:

        x = None
        x_seg = None
        x_channel_seg = None
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
        if channel_seg_image is not None:
            x_channel_seg = get_label_data(channel_seg_image)
            print(x_channel_seg.shape)

        nonlocal _trackmate_objects

        plugin.progress_bar.value = 0
        plugin.progress_bar.show()

        _trackmate_objects = TrackMate(
            xml_path,
            spot_csv_path,
            track_csv_path,
            edges_csv_path,
            AttributeBoxname,
            TrackAttributeBoxname,
            TrackidBox,
            channel_seg_image=x_channel_seg,
            image=x,
            mask=x_mask,
            progress_bar=plugin.progress_bar,
        )

        _refreshStatPlotData()

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

        _Color_tracks(spot_attributes, track_attributes)

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.25080/majora-1b6fd038-014")

    def _refreshTrackData(pred):

        unique_tracks, unique_tracks_properties = pred
        features = {
            "time": map(
                int,
                np.asarray(unique_tracks_properties, dtype="float64")[:, 0],
            ),
            "generation": map(
                int,
                np.asarray(unique_tracks_properties, dtype="float64")[:, 1],
            ),
            "speed": map(
                float,
                np.asarray(unique_tracks_properties, dtype="float64")[:, 2],
            ),
            "directional_change_rate": map(
                float,
                np.asarray(unique_tracks_properties, dtype="float64")[:, 3],
            ),
            "total-intensity": map(
                float,
                np.asarray(unique_tracks_properties, dtype="float64")[:, 4],
            ),
            "volume_pixels": map(
                float,
                np.asarray(unique_tracks_properties, dtype="float64")[:, 5],
            ),
            "acceleration": map(
                float,
                np.asarray(unique_tracks_properties, dtype="float64")[:, 6],
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

        plugin.viewer.value.add_points(vertices, size=2, name="Track_points")

        plugin.viewer.value.add_tracks(
            unique_tracks,
            name="Track",
            features=features,
        )

    def show_fft():

        nonlocal _to_analyze

        fft_plot_class._reset_container(fft_plot_class.scroll_layout)
        if _to_analyze is not None:

            unique_fft_properties = []
            for unique_track_id in _to_analyze:
                (
                    time,
                    xf_sample,
                    ffttotal_sample,
                ) = _trackmate_objects.unique_fft_properties[unique_track_id]
                unique_fft_properties.append(
                    [
                        time,
                        xf_sample,
                        ffttotal_sample,
                    ]
                )
            fft_plot_class._repeat_after_plot()
            plot_ax = fft_plot_class.plot_ax
            plot_ax.cla()

            all_time = []
            all_xf_sample = []
            all_ffttotal_sample = []

            for unique_property in unique_fft_properties:
                (
                    time,
                    xf_sample,
                    ffttotal_sample,
                ) = unique_property

                all_time.append(time)
                all_xf_sample.append(xf_sample)
                all_ffttotal_sample.append(np.ravel(ffttotal_sample))
            max_size = 0
            max_size_index = 0
            for i in range(len(all_ffttotal_sample)):
                size = all_ffttotal_sample[i].shape[0]
                if size > max_size:
                    max_size = size
                    max_size_index = i

            max_all_xf_sample = all_xf_sample[max_size_index]
            resize_all_ffttotal_sample = []
            for i in range(len(all_ffttotal_sample)):
                sample = np.pad(
                    all_ffttotal_sample[i],
                    (
                        0,
                        max_all_xf_sample.shape[0]
                        - all_ffttotal_sample[i].shape[0],
                    ),
                )
                resize_all_ffttotal_sample.append(sample)

            data_plot = pd.DataFrame(
                {
                    "Frequ": max_all_xf_sample,
                    "Amplitude": sum(resize_all_ffttotal_sample),
                }
            )
            sns.lineplot(data_plot, x="Frequ", y="Amplitude", ax=plot_ax)
            plot_ax.set_title("FFT Intensity")
            plot_ax.set_xlabel("Frequency (1/min)")
            plot_ax.set_ylabel("Amplitude")

    def return_color_tracks(pred):

        if not isinstance(pred, int):
            new_seg_image, attribute = pred
            new_seg_image = new_seg_image.astype("uint16")
            for layer in list(plugin.viewer.value.layers):
                if attribute in layer.name:
                    plugin.viewer.value.layers.remove(layer)
            plugin.viewer.value.add_labels(new_seg_image, name=attribute)

    def _Color_tracks(spot_attribute, track_attribute):
        nonlocal _trackmate_objects
        yield 0
        x_seg = get_label_data(plugin_data.seg_image.value)
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
            return_color_tracks(pred)
        return pred

    @magicgui(
        label_head=dict(
            widget_type="Label",
            label=f'<h1> <img src="{kapoorlogo}"> </h1>',
            value=f'<h5><a href=" {citation}"> NapaTrackMater: Track Analysis of TrackMate in Napari</a></h5>',
        ),
        track_model_type=dict(
            widget_type="RadioButtons",
            label="Track Model Type",
            orientation="horizontal",
            choices=track_model_type_choices,
            value=DEFAULTS_MODEL["track_model_type"],
        ),
        track_id_value=dict(widget_type="Label", label="Track ID chosen"),
        track_id_box=dict(
            widget_type="ComboBox",
            visible=True,
            label="Select Track ID to analyze",
            choices=_current_choices,
        ),
        cloud_auto_encoder_model_type=dict(
            widget_type="RadioButtons",
            label="Cloud Auto Encoder Model Type",
            orientation="horizontal",
            choices=cloud_auto_encoder_model_type_choices,
            value=DEFAULTS_MODEL["cloud_auto_encoder_model_type"],
        ),
        cloud_auto_encoder_model=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained Auto Encoder Models",
            choices=models_cloud_auto_encoder,
            value=DEFAULTS_MODEL["model_cloud_auto_encoder"],
        ),
        cloud_auto_encoder_model_none=dict(
            widget_type="Label", visible=False, label="No(Encoder)"
        ),
        model_folder_cloud_auto=dict(
            widget_type="FileEdit",
            visible=False,
            label="Custom Auto Encoder",
            mode="d",
        ),
        cluster_model_type=dict(
            widget_type="RadioButtons",
            label="Cluster Model Type",
            orientation="horizontal",
            choices=cluster_model_type_choices,
            value=DEFAULTS_MODEL["cluster_model_type"],
        ),
        cluster_model=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained Clustering Models",
            choices=models_cluster,
            value=DEFAULTS_MODEL["model_cluster"],
        ),
        cluster_model_none=dict(
            widget_type="Label", visible=False, label="No(Cluster)"
        ),
        model_folder_cluster=dict(
            widget_type="FileEdit",
            visible=False,
            label="Custom Cluster Model",
            mode="d",
        ),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        layout="vertical",
        persist=True,
        call_button=False,
    )
    def plugin(
        viewer: napari.Viewer,
        label_head,
        axes,
        track_model_type,
        track_id_box,
        track_id_value,
        cloud_auto_encoder_model_type,
        cloud_auto_encoder_model,
        cloud_auto_encoder_model_none,
        model_folder_cloud_auto,
        cluster_model_type,
        cluster_model,
        cluster_model_none,
        model_folder_cluster,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:

        if model_selected_cloud_auto_encoder is not None:
            model_cloud_auto_encoder = get_model_cloud_auto_encoder(
                *model_selected_cloud_auto_encoder
            )
        else:
            model_cloud_auto_encoder = None
        if model_selected_cluster is not None:
            model_cluster = get_model_cluster(
                *model_selected_cloud_auto_encoder, *model_selected_cluster
            )
        else:
            model_cluster = None
        print(model_cloud_auto_encoder, model_cluster)

    plugin.label_head.value = '<br>Citation <tt><a href="https://doi.org/10.25080/majora-1b6fd038-014" style="color:gray;">NapaTrackMater Scipy</a></tt>'
    plugin.label_head.native.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
    )
    plugin.progress_bar.hide()

    tabs = QTabWidget()

    data_tab = QWidget()
    _data_tab_layout = QVBoxLayout()
    data_tab.setLayout(_data_tab_layout)
    _data_tab_layout.addWidget(plugin_data.native)
    tabs.addTab(data_tab, "Input Data")

    color_tracks_tab = QWidget()
    _color_tracks_tab_layout = QVBoxLayout()
    color_tracks_tab.setLayout(_color_tracks_tab_layout)
    _color_tracks_tab_layout.addWidget(plugin_color_parameters.native)
    tabs.addTab(color_tracks_tab, "Color Tracks")

    hist_plot_class = TemporalStatistics(tabs)
    hist_plot_tab = hist_plot_class.plot_tab
    tabs.addTab(hist_plot_tab, "Histogram Statistics")

    stat_plot_class = TemporalStatistics(tabs)
    plot_tab = stat_plot_class.plot_tab
    tabs.addTab(plot_tab, "Temporal Statistics")

    fft_plot_class = TemporalStatistics(tabs)
    fft_plot_tab = fft_plot_class.plot_tab
    tabs.addTab(fft_plot_tab, "Phenotype analysis")

    table_tab = Tabulour()
    table_tab.clicked.connect(table_tab._on_user_click)
    tabs.addTab(table_tab, "Table")

    plugin.native.layout().addWidget(tabs)

    def plot_main():

        nonlocal _trackmate_objects
        hist_plot_class._reset_container(hist_plot_class.scroll_layout)
        stat_plot_class._reset_container(stat_plot_class.scroll_layout)

        if _trackmate_objects is not None:
            trackid_key = _trackmate_objects.track_analysis_spot_keys[
                "track_id"
            ]
            key = plugin.track_model_type.value
            for k in _trackmate_objects.AllTrackValues.keys():
                if k is not trackid_key:
                    TrackAttr = []
                    for attr, trackid in zip(
                        _trackmate_objects.AllTrackValues[k],
                        _trackmate_objects.AllTrackValues[trackid_key],
                    ):
                        if key == track_model_type_dict[0]:

                            if (
                                int(trackid)
                                in _trackmate_objects.DividingTrackIds
                            ):

                                TrackAttr.append(float(attr))
                        if key == track_model_type_dict[1]:
                            if (
                                int(trackid)
                                in _trackmate_objects.NormalTrackIds
                            ):
                                TrackAttr.append(float(attr))
                        if key == track_model_type_dict[2]:
                            TrackAttr.append(float(attr))

                    hist_plot_class._repeat_after_plot()
                    hist_ax = hist_plot_class.plot_ax
                    sns.histplot(TrackAttr, kde=True, ax=hist_ax)
                    hist_ax.set_title(str(k))

            if key == track_model_type_dict[0]:

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax
                plot_ax.cla()

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.mitotic_mean_directional_change,
                    _trackmate_objects.mitotic_var_directional_change,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Instantaneous Directional change")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("angle (degrees)")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax
                plot_ax.cla()

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.mitotic_mean_speed,
                    _trackmate_objects.mitotic_var_speed,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Speed")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um/min")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.mitotic_mean_radius,
                    _trackmate_objects.mitotic_var_radius,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Radius")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.mitotic_mean_disp_z,
                    _trackmate_objects.mitotic_var_disp_z,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )

                plot_ax.set_title("Displacement in Z")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.mitotic_mean_disp_y,
                    _trackmate_objects.mitotic_var_disp_y,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )

                plot_ax.set_title("Displacement in Y")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.mitotic_mean_disp_x,
                    _trackmate_objects.mitotic_var_disp_x,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )

                plot_ax.set_title("Displacement in X")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

            if key == track_model_type_dict[1]:

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax
                plot_ax.cla()

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_directional_change,
                    _trackmate_objects.non_mitotic_var_directional_change,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Instantaneous Directional change")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("angle (degrees)")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax
                plot_ax.cla()

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_speed,
                    _trackmate_objects.non_mitotic_var_speed,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Instantaneous Speed")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um/min")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_radius,
                    _trackmate_objects.non_mitotic_var_radius,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Radius")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_disp_z,
                    _trackmate_objects.non_mitotic_var_disp_z,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )

                plot_ax.set_title("Displacement in Z")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_disp_y,
                    _trackmate_objects.non_mitotic_var_disp_y,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )

                plot_ax.set_title("Displacement in Y")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_disp_x,
                    _trackmate_objects.non_mitotic_var_disp_x,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )

                plot_ax.set_title("Displacement in X")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

            if key == track_model_type_dict[2]:

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax
                plot_ax.cla()

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.all_mean_directional_change,
                    _trackmate_objects.all_var_directional_change,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Instantaneous Directional change")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("angle (degrees)")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax
                plot_ax.cla()

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_speed,
                    _trackmate_objects.non_mitotic_var_speed,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Instantaneous  Speed")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um/min")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_radius,
                    _trackmate_objects.non_mitotic_var_radius,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Radius")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_disp_z,
                    _trackmate_objects.non_mitotic_var_disp_z,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )

                plot_ax.set_title("Displacement in Z")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_disp_y,
                    _trackmate_objects.non_mitotic_var_disp_y,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )

                plot_ax.set_title("Displacement in Y")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")

                stat_plot_class._repeat_after_plot()
                plot_ax = stat_plot_class.plot_ax

                plot_ax.errorbar(
                    _trackmate_objects.time,
                    _trackmate_objects.non_mitotic_mean_disp_x,
                    _trackmate_objects.non_mitotic_var_disp_x,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )

                plot_ax.set_title("Displacement in X")
                plot_ax.set_xlabel("Time (min)")
                plot_ax.set_ylabel("um")
            select_track_nature()
            for layer in list(plugin.viewer.value.layers):
                if isinstance(layer, napari.layers.Tracks):
                    table_tab.layer = layer

    def _refreshStatPlotData():
        nonlocal _trackmate_objects, _current_choices, _dividing_choices, _normal_choices, _both_choices, _dividing_track_ids_analyze, _normal_track_ids_analyze, _both_track_ids_analyze
        plugin.progress_bar.label = "Analyzing Tracks"
        columns = None
        root_cells = None
        root_spots = _trackmate_objects.root_spots
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

        plugin_color_parameters.track_attributes.choices = (
            _trackmate_objects.TrackAttributeids
        )
        plugin_color_parameters.spot_attributes.choices = (
            _trackmate_objects.Attributeids
        )
        plugin.progress_bar.label = "Creating Table"
        plugin.progress_bar.range = (0, len(root_spots) - 1)

        v = next(iter(root_spots.values()))
        columns = [value for value in v.keys()]
        for count, (k, v) in enumerate(root_spots.items()):

            plugin.progress_bar.value = count
            float_list = _analyze_tracks(v)
            if root_cells is None:
                root_cells = np.asarray(float_list)
            else:
                root_cells = np.vstack((root_cells, np.asarray(float_list)))

        print(f"Making pandas dataframe  {root_cells.shape}")
        columns[0] = "Root_Cell_ID"
        colindex = 0
        for i in range(len(columns)):
            col = columns[i]
            if col == id_key:
                colindex = i
        df = pd.DataFrame(
            root_cells,
            columns=columns,
            dtype=object,
        )
        df = df_column_switch(df, columns[0], columns[colindex])
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

    def _analyze_tracks(v):
        float_list = list(v.values())
        return float_list

    def df_column_switch(df, column1, column2):
        i = list(df.columns)
        a, b = i.index(column1), i.index(column2)
        i[b], i[a] = i[a], i[b]
        df = df[i]
        return df

    def select_track_nature():
        key = plugin.track_model_type.value
        nonlocal _trackmate_objects, _track_ids_analyze, _dividing_track_ids_analyze, _normal_track_ids_analyze, _both_track_ids_analyze, _current_choices, _to_analyze
        if _trackmate_objects is not None:
            if key == track_model_type_dict[0]:
                plugin.track_id_box.choices = _dividing_choices
                _track_ids_analyze = _dividing_track_ids_analyze
                _current_choices = _dividing_choices
            if key == track_model_type_dict[1]:
                plugin.track_id_box.choices = _normal_choices
                _track_ids_analyze = _normal_track_ids_analyze
                _current_choices = _normal_choices
            if key == track_model_type_dict[2]:
                plugin.track_id_box.choices = _both_choices
                _track_ids_analyze = _both_track_ids_analyze
                _current_choices = _both_choices

            _track_ids_analyze = list(map(int, _track_ids_analyze))
            _to_analyze = _track_ids_analyze

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet(
                "" if valid else "background-color: red"
            )

    def show_track(track_id):

        nonlocal _track_ids_analyze, _to_analyze
        unique_tracks = []
        unique_tracks_properties = []

        if str(track_id) not in TrackidBox and track_id is not None:
            _to_analyze = [int(track_id)]
        else:
            _to_analyze = _track_ids_analyze
        if _to_analyze is not None:
            show_fft()
            for unique_track_id in _to_analyze:

                tracklets = _trackmate_objects.unique_tracks[unique_track_id]
                tracklets_properties = (
                    _trackmate_objects.unique_track_properties[unique_track_id]
                )

                unique_tracks.append(tracklets)
                unique_tracks_properties.append(tracklets_properties)

            unique_tracks = np.concatenate(unique_tracks, axis=0)
            unique_tracks_properties = np.concatenate(
                unique_tracks_properties, axis=0
            )
            pred = unique_tracks, unique_tracks_properties
            _refreshTrackData(pred)
            select_track_nature()

    @change_handler(plugin.track_id_box, init=False)
    def _track_id_box_change(value):

        plugin.track_id_box.value = value
        plugin.track_id_value.value = value

        nonlocal _track_ids_analyze, _trackmate_objects
        if (
            _trackmate_objects is not None
            and _track_ids_analyze is not None
            and value is not None
        ):

            track_id = value
            show_track(track_id)

    widget_for_cloud_auto_encoder_modeltype = {
        CloudAutoEncoder: plugin.cloud_auto_encoder_model,
        "No(Encoder)": plugin.cloud_auto_encoder_model_none,
        CUSTOM_MODEL_CLOUD_AUTO_ENCODER: plugin.model_folder_cloud_auto,
    }

    @change_handler(plugin.cloud_auto_encoder_model_type, init=False)
    def _cloud_auto_encoder_model_type_change(
        cloud_auto_encoder_model_type: Union[str, type]
    ):
        selected = widget_for_cloud_auto_encoder_modeltype[
            cloud_auto_encoder_model_type
        ]
        for w in {
            plugin.cloud_auto_encoder_model,
            plugin.cloud_auto_encoder_model_none,
            plugin.model_folder_cloud_auto,
        } - {selected}:
            w.hide()

        selected.show()

        # Trigger model change
        selected.changed(selected.value)

    widget_for_cluster_modeltype = {
        DeepEmbeddedClustering: plugin.cluster_model,
        "No(Cluster)": plugin.cluster_model_none,
        CUSTOM_MODEL_CLUSTER: plugin.model_folder_cluster,
    }

    @change_handler(plugin.cluster_model_type, init=False)
    def _cluster_model_type_change(cluster_model_type: Union[str, type]):
        selected = widget_for_cluster_modeltype[cluster_model_type]
        for w in {
            plugin.cluster_model,
            plugin.cluster_model_none,
            plugin.model_folder_cluster,
        } - {selected}:
            w.hide()

        selected.show()

        # Trigger model change
        selected.changed(selected.value)

    @change_handler(plugin.track_model_type, init=False)
    def _change_track_model_type(value):

        plugin.track_model_type.value = value
        select_track_nature()
        plot_main()
        show_fft()

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

    @change_handler(plugin_data.image, init=False)
    def _image_change(image: napari.layers.Image):
        plugin_data.image.tooltip = (
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
        if axes == plugin_data.axes.value:
            # make sure to trigger a changed event, even if value didn't actually change
            plugin_data.axes.changed(axes)
        else:
            plugin_data.axes.value = axes

    # -> triggered by _image_change
    @change_handler(plugin_data.axes, init=False)
    def _axes_change():
        value = plugin_data.axes.value
        print(f"axes is {value}")

    return plugin
