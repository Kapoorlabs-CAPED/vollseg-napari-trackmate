"""
VollSeg Napari Track .
Made by Kapoorlabs, 2022
"""

import functools
import math
import os
import time
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
from napari.qt import thread_worker
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget
from scipy import spatial


def plugin_wrapper_track():

    from napatrackmater import (
        CloudAutoEncoder,
        DeepEmbeddedClustering,
        load_json,
    )
    from napatrackmater.pretrained import (
        get_model_folder,
        get_registered_models,
    )
    from napatrackmater.Trackmate import TrackMate
    from skimage.util import map_array

    from vollseg_napari_trackmate._temporal_plots import TemporalStatistics

    DEBUG = False

    def abspath(root, relpath):
        root = Path(root)
        if root.is_dir():
            path = root / relpath
        else:
            path = root.parent / relpath
        return str(path.absolute())

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.25080/majora-1b6fd038-014")
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
    clicked_location = None
    track_centroid_tree = None
    track_centroid_list = None

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
    model_cluster_configs = dict()

    model_selected_cloud_auto_encoder = None
    model_selected_cluster = None
    size_catagories_json = None

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
    DEFAULTS_PARAMETERS = dict(batch_size=8, step_size=10)

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

    @functools.lru_cache(maxsize=None)
    def get_model_cloud_auto_encoder(
        cloud_auto_encoder_model_type, model_cloud_auto_encoder
    ):
        if cloud_auto_encoder_model_type == CUSTOM_MODEL_CLOUD_AUTO_ENCODER:
            path_auto = Path(model_cloud_auto_encoder)
            path_auto.is_file() or _raise(
                FileNotFoundError(f"{path_auto} is not a file")
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
                os.path.join(path_auto.parent, path_auto.stem + ".pt")
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

        if autoencoder is not None:
            if cluster_model_type == CUSTOM_MODEL_CLUSTER:
                path_cluster = Path(model_cluster)
                path_cluster.is_file() or _raise(
                    FileNotFoundError(f"{path_cluster} is not a file")
                )

                checkpoint = torch.load(
                    os.path.join(
                        path_cluster.parent, path_cluster.stem + ".pt"
                    )
                )
                num_clusters = checkpoint["model_state_dict"][
                    "clustering_layer.weight"
                ].shape[0]

                model = DeepEmbeddedClustering(
                    autoencoder=autoencoder, num_clusters=num_clusters
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                return model

            elif cluster_model_type != DEFAULTS_MODEL["model_cluster_none"]:
                return cluster_model_type.local_from_pretrained(
                    model_cluster, autoencoder
                )
            else:
                return None

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
            mode="r",
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
            mode="r",
        ),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        layout="vertical",
        persist=True,
        call_button=False,
    )
    def plugin(
        viewer: napari.Viewer,
        label_head,
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

        pass

    @plugin.viewer.value.mouse_double_click_callbacks.append
    def get_event(viewer, event):
        nonlocal clicked_location
        clicked_location = event.position
        print("Location clicked", clicked_location)
        if track_centroid_list is not None:
            if len(track_centroid_list) > 0:
                dist, index = track_centroid_tree.query(clicked_location)
                nearest_track_location = track_centroid_list[index]
                nearest_track_id = _trackmate_objects.unique_track_centroid[
                    nearest_track_location
                ]
                print(
                    "nearest track id found",
                    nearest_track_id,
                    "showing display",
                )
                show_track(nearest_track_id)

    class Updater_Auto_Encoder:
        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace

            self.debug = debug
            self.valid = SimpleNamespace(
                **{k: False for k in ("model_autoencoder",)}
            )
            self.args = SimpleNamespace()
            self.viewer = None

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args, k, args)
            self._update()

        def help(self, msg):
            if self.viewer is not None:
                self.viewer.help = msg
            elif len(str(msg)) > 0:
                print(f"HELP: {msg}")

        def _update(self):

            if self.viewer is None:

                if plugin.viewer.value is not None:
                    self.viewer = plugin.viewer.value

            def _model(valid):

                widgets_valid(
                    plugin.cloud_auto_encoder_model,
                    plugin.model_folder_cloud_auto,
                    valid=valid,
                )

            def _restore():
                widgets_valid(
                    plugin.image, valid=plugin.image.value is not None
                )

            all_valid = False
            for layer in list(plugin.viewer.value.layers):
                if isinstance(layer, napari.layers.Labels):
                    all_valid = True
                    break
            help_msg = ""

            if self.valid.model_autoencoder:

                widgets_valid(
                    plugin.cloud_auto_encoder_model,
                    plugin.model_folder_cloud_auto.line_edit,
                    valid=all_valid,
                )

            else:

                _model(self.valid.model_autoencoder)

                _restore()
            self.help(help_msg)

    class Updater_Cluster:
        def __init__(self, debug=DEBUG):
            from types import SimpleNamespace

            self.debug = debug
            self.valid = SimpleNamespace(
                **{k: False for k in ("model_autoencoder",)}
            )
            self.args = SimpleNamespace()
            self.viewer = None

        def __call__(self, k, valid, args=None):
            assert k in vars(self.valid)
            setattr(self.valid, k, bool(valid))
            setattr(self.args, k, args)
            self._update()

        def help(self, msg):
            if self.viewer is not None:
                self.viewer.help = msg
            elif len(str(msg)) > 0:
                print(f"HELP: {msg}")

        def _update(self):

            if self.viewer is None:

                if plugin.viewer.value is not None:
                    self.viewer = plugin.viewer.value

            def _model(valid):

                widgets_valid(
                    plugin.cluster_model,
                    plugin.model_folder_cluster,
                    valid=valid,
                )

            def _restore():
                widgets_valid(
                    plugin.image, valid=plugin.image.value is not None
                )

            all_valid = False
            for layer in list(plugin.viewer.value.layers):
                if isinstance(layer, napari.layers.Labels):
                    all_valid = True
                    break
            help_msg = ""

            if self.valid.model_autoencoder:

                widgets_valid(
                    plugin.cluster_model,
                    plugin.model_folder_cluster.line_edit,
                    valid=all_valid,
                )

            else:

                _model(self.valid.model_autoencoder)

                _restore()
            self.help(help_msg)

    update_cloud_auto_encoder = Updater_Auto_Encoder()
    update_cluster = Updater_Cluster()

    def select_model_cloud_auto_encoder(key):
        nonlocal model_selected_cloud_auto_encoder
        if key is not None:
            model_selected_cloud_auto_encoder = key
            # config_cloud_auto_encoder = model_cloud_auto_encoder_configs.get(
            #    key
            # )
            update_cloud_auto_encoder("model_autoencoder", True)
        if (
            plugin.cloud_auto_encoder_model_type.value
            == DEFAULTS_MODEL["model_cloud_auto_encoder_none"]
        ):
            model_selected_cloud_auto_encoder = None

    def select_model_cluster(key):
        nonlocal model_selected_cluster
        if key is not None:
            model_selected_cluster = key
            # config_cloud_auto_encoder = model_cloud_auto_encoder_configs.get(
            #    key
            # )
            update_cluster("model_autoencoder", True)
        if (
            plugin.cluster_model_type.value
            == DEFAULTS_MODEL["model_cluster_none"]
        ):
            model_selected_cluster = None

    @change_handler(
        plugin.cluster_model,
        plugin.cluster_model_none,
        init=False,
    )
    def _model_change_cluster(model_name_cluster: str):

        if Signal.sender() is not plugin.cluster_model_none:
            model_class_cluster = DeepEmbeddedClustering

            if model_class_cluster is not None:
                if Signal.sender is not None:
                    model_name = model_name_cluster
                elif plugin.cluster_model.value is not None:
                    model_name = plugin.cluster_model.value

                key_cluster = (
                    model_class_cluster,
                    model_name,
                )
                if key_cluster not in model_cluster_configs:

                    @thread_worker
                    def _get_model_folder():
                        return get_model_folder(*key_cluster)

                    def _process_model_folder(rpath):

                        path = rpath[0]
                        model_cluster_configs[key_cluster] = str(path)
                        select_model_cluster(key_cluster)
                        plugin.progress_bar.hide()

                    worker = _get_model_folder()
                    worker.returned.connect(_process_model_folder)
                    worker.start()

                    # delay showing progress bar -> won't show up if model already downloaded
                    # TODO: hacky -> better way to do this?
                    time.sleep(0.1)
                    plugin.progress_bar.label = (
                        "Downloading Auto Encoder model"
                    )
                    plugin.progress_bar.show()

                else:
                    select_model_cluster(key_cluster)
        else:
            select_model_cluster(None)

            plugin.model_folder_cluster.line_edit.tooltip = (
                "Invalid model file"
            )

    @change_handler(
        plugin.cloud_auto_encoder_model,
        plugin.cloud_auto_encoder_model_none,
        init=False,
    )
    def _model_change_cloud_auto_encoder(model_name_cloud_auto_encoder: str):

        if Signal.sender() is not plugin.cloud_auto_encoder_model_none:
            model_class_cloud_auto_encoder = CloudAutoEncoder

            if model_class_cloud_auto_encoder is not None:
                if Signal.sender is not None:
                    model_name = model_name_cloud_auto_encoder
                elif plugin.cloud_auto_encoder_model.value is not None:
                    model_name = plugin.cloud_auto_encoder_model.value

                key_cloud_auto_encoder = (
                    model_class_cloud_auto_encoder,
                    model_name,
                )
                if (
                    key_cloud_auto_encoder
                    not in model_cloud_auto_encoder_configs
                ):

                    @thread_worker
                    def _get_model_folder():
                        return get_model_folder(*key_cloud_auto_encoder)

                    def _process_model_folder(rpath):

                        path = rpath[0]
                        model_cloud_auto_encoder_configs[
                            key_cloud_auto_encoder
                        ] = load_json(
                            str(
                                os.path.join(
                                    os.path.join(
                                        path.parent.as_posix(), path.name
                                    ),
                                    model_name + ".json",
                                )
                            )
                        )

                        select_model_cloud_auto_encoder(key_cloud_auto_encoder)
                        plugin.progress_bar.hide()

                    worker = _get_model_folder()
                    worker.returned.connect(_process_model_folder)
                    worker.start()

                    # delay showing progress bar -> won't show up if model already downloaded
                    # TODO: hacky -> better way to do this?
                    time.sleep(0.1)
                    plugin.progress_bar.label = (
                        "Downloading Auto Encoder model"
                    )
                    plugin.progress_bar.show()

                else:
                    select_model_cloud_auto_encoder(key_cloud_auto_encoder)
        else:
            select_model_cloud_auto_encoder(None)

            plugin.model_folder_cloud_auto.line_edit.tooltip = (
                "Invalid model directory"
            )

    @change_handler(plugin.model_folder_cloud_auto, init=False)
    def _model_cloud_auto_folder_change(_path: str):
        path = Path(_path)
        key = CUSTOM_MODEL_CLOUD_AUTO_ENCODER, path
        try:
            if not path.is_file():
                return
            model_cloud_auto_encoder_configs[key] = load_json(
                str(os.path.join(path.parent.as_posix(), path.stem + ".json"))
            )
        except FileNotFoundError:
            print(f"file not found in {path.parent}, {path.stem}]")
        finally:
            select_model_cloud_auto_encoder(key)

    @change_handler(plugin.model_folder_cluster, init=False)
    def _model_cluster_folder_change(_path: str):
        path = Path(_path)
        key = CUSTOM_MODEL_CLUSTER, path
        try:
            if not path.is_file():
                return

        except FileNotFoundError:
            pass
        finally:
            select_model_cluster(key)

    _track_ids_analyze = None
    _to_analyze = None
    _trackmate_objects = None

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
        master_xml_path=dict(
            widget_type="FileEdit",
            visible=True,
            label="NapaTrackMater Master xml",
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
        size_catagories=dict(
            widget_type="FileEdit",
            visible=True,
            label="Sentinal size catagories json",
            mode="r",
        ),
        axes=dict(
            widget_type="LineEdit",
            label="Image Axes",
            value=DEFAULTS_MODEL["axes"],
        ),
        batch_size=dict(
            widget_type="SpinBox",
            label="Batch size (clustering model)",
            min=1,
            max=10000000,
            step=1,
            value=DEFAULTS_PARAMETERS["batch_size"],
        ),
        plot_step_size=dict(
            widget_type="SpinBox",
            label="Cluster class time step",
            min=1,
            max=10000000,
            step=1,
            value=DEFAULTS_PARAMETERS["step_size"],
        ),
        compute_button=dict(widget_type="PushButton", text="Compute"),
        layout="vertical",
        persist=False,
        call_button=False,
    )
    def plugin_data(
        image: Union[napari.layers.Image, None],
        seg_image: Union[napari.layers.Labels, None],
        channel_seg_image: Union[napari.layers.Labels, None],
        mask_image: Union[napari.layers.Labels, None],
        xml_path,
        master_xml_path,
        track_csv_path,
        spot_csv_path,
        edges_csv_path,
        size_catagories,
        axes,
        batch_size,
        plot_step_size,
        compute_button,
    ) -> List[napari.types.LayerDataTuple]:

        pass

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

    def _refreshTrackData(pred):

        nonlocal _to_analyze
        unique_tracks, unique_tracks_properties, track_id = pred
        features = {
            "time": np.asarray(unique_tracks_properties, dtype="float64")[
                :, 0
            ],
            "generation": np.asarray(
                unique_tracks_properties, dtype="float64"
            )[:, 2],
            "speed": np.asarray(unique_tracks_properties, dtype="float64")[
                :, 3
            ],
            "directional_change_rate": np.asarray(
                unique_tracks_properties, dtype="float64"
            )[:, 4],
            "total-intensity": np.asarray(
                unique_tracks_properties, dtype="float64"
            )[:, 5],
            "volume_pixels": np.asarray(
                unique_tracks_properties, dtype="float64"
            )[:, 6],
            "acceleration": np.asarray(
                unique_tracks_properties, dtype="float64"
            )[:, 7],
            "cluster_class": np.asarray(
                unique_tracks_properties, dtype="float64"
            )[:, 8],
            "cluster_score": np.asarray(
                unique_tracks_properties, dtype="float64"
            )[:, 9],
        }
        print("Refreshing track data")
        for layer in list(plugin.viewer.value.layers):
            if (
                "Track" == layer.name
                or "Boxes" == layer.name
                or "Track_points" == layer.name
            ):
                plugin.viewer.value.layers.remove(layer)
        vertices = unique_tracks[:, 1:]
        plugin.viewer.value.add_points(vertices, name="Track_points", size=1)
        print("Added vertices")
        plugin.viewer.value.add_tracks(
            unique_tracks,
            name="Track",
            features=features,
        )
        print("Track data refreshed")
        if str(track_id) not in TrackidBox and track_id is not None:
            _to_analyze = [int(track_id)]
        show_phenotype()
        select_track_nature()

    def show_phenotype():

        nonlocal _to_analyze, size_catagories_json

        phenotype_plot_class._reset_container(
            phenotype_plot_class.scroll_layout
        )
        if _to_analyze is not None:

            unique_fft_properties = []
            unique_shape_properties = []
            phenotype_plot_class._repeat_after_plot()
            plot_ax = phenotype_plot_class.plot_ax
            plot_ax.cla()

            for unique_track_id in _to_analyze:

                for k in _trackmate_objects.unique_fft_properties[
                    unique_track_id
                ].keys():

                    unique_fft_properties_tracklet = (
                        _trackmate_objects.unique_fft_properties[
                            unique_track_id
                        ][k]
                    )
                    unique_shape_properties_tracklet = (
                        _trackmate_objects.unique_shape_properties[
                            unique_track_id
                        ][k]
                    )
                    (
                        time,
                        intensity,
                        xf_sample,
                        ffttotal_sample,
                    ) = unique_fft_properties_tracklet

                    (
                        cluster_time,
                        cluster_radius,
                        cluster_volume,
                        cluster_eccentricity_x,
                        cluster_eccentricity_y,
                        cluster_eccentricity_z,
                        cluster_surface_area,
                        cluster_class,
                        cluster_class_score,
                    ) = unique_shape_properties_tracklet
                    cluster_class_name = []
                    if size_catagories_json is not None:
                        for i in range(cluster_class.shape[0]):
                            if cluster_class[i] is not None:
                                cluster_class_name.append(
                                    size_catagories_json[
                                        str(int(cluster_class[i]))
                                    ]
                                )
                            else:
                                cluster_class_name.append(None)

                    unique_fft_properties.append(
                        [
                            time,
                            intensity,
                            xf_sample,
                            ffttotal_sample,
                        ]
                    )
                    if size_catagories_json is None:

                        unique_shape_properties.append(
                            [
                                cluster_time,
                                cluster_radius,
                                cluster_volume,
                                cluster_eccentricity_x,
                                cluster_eccentricity_y,
                                cluster_eccentricity_z,
                                cluster_surface_area,
                                cluster_class,
                                cluster_class_score,
                            ]
                        )

                    if size_catagories_json is not None:

                        unique_shape_properties.append(
                            [
                                cluster_time,
                                cluster_radius,
                                cluster_volume,
                                cluster_eccentricity_x,
                                cluster_eccentricity_y,
                                cluster_eccentricity_z,
                                cluster_surface_area,
                                cluster_class,
                                cluster_class_score,
                                cluster_class_name,
                            ]
                        )

            global_data_cluster_plot = []
            for i in range(len(unique_shape_properties)):

                print("in", i)
                current_unique_shape_properties = unique_shape_properties[i]
                cluster_time = current_unique_shape_properties[0]
                cluster_radius = current_unique_shape_properties[1]
                cluster_volume = current_unique_shape_properties[2]
                cluster_eccentricity_x = current_unique_shape_properties[3]
                cluster_eccentricity_y = current_unique_shape_properties[4]
                cluster_eccentricity_z = current_unique_shape_properties[5]
                cluster_surface_area = current_unique_shape_properties[6]
                cluster_class = current_unique_shape_properties[7]
                cluster_class_score = current_unique_shape_properties[8]
                if size_catagories_json is not None:

                    cluster_class_name = current_unique_shape_properties[9]
                    data_cluster_plot = pd.DataFrame(
                        {
                            "Time": cluster_time,
                            "Radius": cluster_radius,
                            "Volume": cluster_volume,
                            "Eccentricity_X": cluster_eccentricity_x,
                            "Eccentricity_Y": cluster_eccentricity_y,
                            "Eccentricity_Z": cluster_eccentricity_z,
                            "Surface_Area": cluster_surface_area,
                            "Class": cluster_class,
                            "Class_Score": cluster_class_score,
                            "Class_Name": cluster_class_name,
                        }
                    )
                else:
                    data_cluster_plot = pd.DataFrame(
                        {
                            "Time": cluster_time,
                            "Radius": cluster_radius,
                            "Volume": cluster_volume,
                            "Eccentricity_X": cluster_eccentricity_x,
                            "Eccentricity_Y": cluster_eccentricity_y,
                            "Eccentricity_Z": cluster_eccentricity_z,
                            "Surface_Area": cluster_surface_area,
                            "Class": cluster_class,
                            "Class_Score": cluster_class_score,
                        }
                    )
                data_cluster_plot.set_index(
                    np.ones(np.asarray(cluster_time).shape) * i
                )
                data_cluster_plot = data_cluster_plot.mask(
                    data_cluster_plot.astype(object).eq("None")
                ).dropna()

                if len(global_data_cluster_plot) == 0:
                    global_data_cluster_plot = data_cluster_plot
                else:
                    global_data_cluster_plot = pd.concat(
                        [global_data_cluster_plot, data_cluster_plot],
                        ignore_index=False,
                    )
            print(print(global_data_cluster_plot.index))
            if size_catagories_json is None:
                sns.lineplot(
                    global_data_cluster_plot,
                    x="Time",
                    y="Class",
                    ax=plot_ax,
                )
            if size_catagories_json is not None:
                sns.lineplot(
                    global_data_cluster_plot,
                    x="Time",
                    y="Class_Name",
                    ax=plot_ax,
                )
            plot_ax.set_title("Cluster class")
            plot_ax.set_xlabel("Time (min)")

            phenotype_plot_class._repeat_after_plot()
            plot_ax = phenotype_plot_class.plot_ax

            sns.lineplot(
                global_data_cluster_plot,
                x="Time",
                y="Radius",
                ax=plot_ax,
            )
            plot_ax.set_title("Radius")
            plot_ax.set_xlabel("Time (min)")

            phenotype_plot_class._repeat_after_plot()
            plot_ax = phenotype_plot_class.plot_ax

            sns.lineplot(
                global_data_cluster_plot,
                x="Time",
                y="Volume",
                ax=plot_ax,
            )
            plot_ax.set_title("Volume")
            plot_ax.set_xlabel("Time (min)")

            phenotype_plot_class._repeat_after_plot()
            plot_ax = phenotype_plot_class.plot_ax

            sns.lineplot(
                global_data_cluster_plot,
                x="Time",
                y="Surface_Area",
                ax=plot_ax,
            )
            plot_ax.set_title("Surface_Area")
            plot_ax.set_xlabel("Time (min)")

            phenotype_plot_class._repeat_after_plot()
            plot_ax = phenotype_plot_class.plot_ax

            sns.lineplot(
                global_data_cluster_plot,
                x="Time",
                y="Eccentricity_X",
                ax=plot_ax,
            )
            plot_ax.set_title("Eccentricity X")
            plot_ax.set_xlabel("Time (min)")

            phenotype_plot_class._repeat_after_plot()
            plot_ax = phenotype_plot_class.plot_ax

            sns.lineplot(
                global_data_cluster_plot,
                x="Time",
                y="Eccentricity_Y",
                ax=plot_ax,
            )
            plot_ax.set_title("Eccentricity Y")
            plot_ax.set_xlabel("Time (min)")

            phenotype_plot_class._repeat_after_plot()
            plot_ax = phenotype_plot_class.plot_ax

            sns.lineplot(
                global_data_cluster_plot,
                x="Time",
                y="Eccentricity_Z",
                ax=plot_ax,
            )
            plot_ax.set_title("Eccentricity Z")
            plot_ax.set_xlabel("Time (min)")

            data_fft_plot = pd.DataFrame(
                {
                    "Frequ": unique_fft_properties[0][2],
                    "Amplitude": np.sum(unique_fft_properties, axis=0)[3],
                }
            )

            data_time_plot = pd.DataFrame(
                {
                    "Time": unique_fft_properties[0][0],
                    "Intensity": np.sum(unique_fft_properties, axis=0)[1],
                }
            )
            phenotype_plot_class._repeat_after_plot()
            plot_ax = phenotype_plot_class.plot_ax

            sns.lineplot(data_time_plot, x="Time", y="Intensity", ax=plot_ax)
            plot_ax.set_title("Cell Intensity")
            plot_ax.set_xlabel("Time (min)")
            plot_ax.set_ylabel("Amplitude")

            phenotype_plot_class._repeat_after_plot()
            plot_ax = phenotype_plot_class.plot_ax

            sns.lineplot(data_fft_plot, x="Frequ", y="Amplitude", ax=plot_ax)
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

    phenotype_plot_class = TemporalStatistics(tabs)
    fft_plot_tab = phenotype_plot_class.plot_tab
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

                if model_selected_cloud_auto_encoder is not None:

                    data_columns = ["Time", "Mitotic_Cluster_Class"]
                    data = []

                    for i in range(
                        0,
                        len(_trackmate_objects.mitotic_cluster_class),
                        plugin_data.plot_step_size.value,
                    ):
                        time = _trackmate_objects.time[i]

                        class_array = _trackmate_objects.mitotic_cluster_class[
                            i
                        ]
                        for i in range(class_array.shape[0]):
                            data.append([time, class_array[i]])
                    clusters = pd.DataFrame(data, columns=data_columns)

                    sns.violinplot(
                        x="Time",
                        y="Mitotic_Cluster_Class",
                        data=clusters,
                        ax=plot_ax,
                    )

                    plot_ax.set_xticklabels([])

                    plot_ax.set_xlabel("Time (min)")
                    plot_ax.set_ylabel("Class")

                    stat_plot_class._repeat_after_plot()
                    plot_ax = stat_plot_class.plot_ax

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
                    _trackmate_objects.mitotic_mean_acc,
                    _trackmate_objects.mitotic_var_acc,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Acceleration")
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

                if model_selected_cloud_auto_encoder is not None:

                    data_columns = ["Time", "Non_Mitotic_Cluster_Class"]
                    data = []

                    for i in range(
                        0,
                        len(_trackmate_objects.non_mitotic_cluster_class),
                        plugin_data.plot_step_size.value,
                    ):
                        time = _trackmate_objects.time[i]

                        class_array = (
                            _trackmate_objects.non_mitotic_cluster_class[i]
                        )
                        for i in range(class_array.shape[0]):
                            data.append([time, class_array[i]])
                    clusters = pd.DataFrame(data, columns=data_columns)
                    sns.violinplot(
                        x="Time",
                        y="Non_Mitotic_Cluster_Class",
                        data=clusters,
                        ax=plot_ax,
                    )

                    plot_ax.set_xticklabels([])
                    plot_ax.set_xlabel("Time (min)")
                    plot_ax.set_ylabel("Class")

                    stat_plot_class._repeat_after_plot()
                    plot_ax = stat_plot_class.plot_ax

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
                    _trackmate_objects.mitotic_mean_acc,
                    _trackmate_objects.mitotic_var_acc,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Acceleration")
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

                if model_selected_cloud_auto_encoder is not None:

                    data_columns = ["Time", "All_CellType_Cluster_Class"]
                    data = []

                    for i in range(
                        0,
                        len(_trackmate_objects.all_cluster_class),
                        plugin_data.plot_step_size.value,
                    ):
                        time = _trackmate_objects.time[i]

                        class_array = _trackmate_objects.all_cluster_class[i]
                        for i in range(class_array.shape[0]):
                            data.append([time, class_array[i]])
                    clusters = pd.DataFrame(data, columns=data_columns)
                    sns.violinplot(
                        x="Time",
                        y="All_CellType_Cluster_Class",
                        data=clusters,
                        ax=plot_ax,
                    )
                    plot_ax.set_xticklabels([])
                    plot_ax.set_xlabel("Time (min)")
                    plot_ax.set_ylabel("Class")

                    stat_plot_class._repeat_after_plot()
                    plot_ax = stat_plot_class.plot_ax

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
                    _trackmate_objects.mitotic_mean_acc,
                    _trackmate_objects.mitotic_var_acc,
                    linestyle="None",
                    marker=".",
                    mfc="green",
                    ecolor="green",
                )
                plot_ax.set_title("Acceleration")
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
        show_phenotype()

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
            if key == track_model_type_dict[1]:
                plugin.track_id_box.choices = _normal_choices
                _track_ids_analyze = _normal_track_ids_analyze
            if key == track_model_type_dict[2]:
                plugin.track_id_box.choices = _both_choices
                _track_ids_analyze = _both_track_ids_analyze

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

        if str(track_id) not in TrackidBox and track_id is not None:
            _to_analyze = [int(track_id)]
        else:
            _to_analyze = _track_ids_analyze

        if _to_analyze is not None:

            unique_tracks = np.concatenate(
                [
                    _trackmate_objects.unique_tracks[unique_track_id]
                    for unique_track_id in _to_analyze
                ]
            )
            unique_tracks_properties = np.concatenate(
                [
                    _trackmate_objects.unique_track_properties[unique_track_id]
                    for unique_track_id in _to_analyze
                ]
            )

            pred = unique_tracks, unique_tracks_properties, track_id

            _refreshTrackData(pred)

    @change_handler(plugin_data.batch_size)
    def _batch_size_change(value: int):

        plugin_data.compute_button.enabled = True
        plugin_data.batch_size.value = value

    @change_handler(plugin_data.plot_step_size)
    def _plot_step_size_change(value: int):

        plugin_data.compute_button.enabled = True
        plugin_data.plot_step_size.value = value

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
        plugin_data.compute_button.enabled = True
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

    plugin_data.compute_button.native.setStyleSheet("background-color: orange")

    @change_handler(plugin_data.compute_button)
    def _compute():

        _actual_computer()

    def _actual_computer():
        x = None
        x_seg = None
        x_channel_seg = None
        x_mask = None

        nonlocal size_catagories_json

        if (
            plugin_data.size_catagories is not None
            and plugin_data.size_catagories.value.is_file()
        ):

            size_catagories_json = load_json(plugin_data.size_catagories.value)

        if plugin_data.xml_path.value is not None:
            save_dir = os.path.join(
                plugin_data.xml_path.value.parent.as_posix(), "runs"
            )
            Path(save_dir).mkdir(exist_ok=True)
        else:
            save_dir = None
        if plugin_data.image.value is not None:
            x = get_data(plugin_data.image.value)
            print(x.shape)

        if plugin_data.seg_image.value is not None:
            x_seg = get_label_data(plugin_data.seg_image.value)
            print(x_seg.shape)
        if plugin_data.mask_image.value is not None:
            x_mask = get_label_data(plugin_data.mask_image.value)
            print(x_mask.shape)
        if plugin_data.channel_seg_image.value is not None:
            x_channel_seg = get_label_data(plugin_data.channel_seg_image.value)
            print(x_channel_seg.shape)

        nonlocal _trackmate_objects

        if model_selected_cluster is not None:
            model_cluster = get_model_cluster(
                *model_selected_cloud_auto_encoder,
                *model_selected_cluster,
            )

            try:
                device = torch.device("cuda:0")
            except ValueError:
                device = torch.device("cpu")
            model_cluster.to(device)
        else:
            model_cluster = None

        plugin.progress_bar.value = 0
        plugin.progress_bar.show()
        num_points = 0
        if model_selected_cloud_auto_encoder is not None:
            (
                cloud_auto_encoder_model_type,
                model_cloud_auto_encoder,
            ) = model_selected_cloud_auto_encoder
            config = model_cloud_auto_encoder_configs[
                (cloud_auto_encoder_model_type, model_cloud_auto_encoder)
            ]
            if len(config) > 0:
                num_points = config["num_points"]
            else:
                num_points = 0

        _trackmate_objects = TrackMate(
            plugin_data.xml_path.value,
            plugin_data.spot_csv_path.value,
            plugin_data.track_csv_path.value,
            plugin_data.edges_csv_path.value,
            AttributeBoxname,
            TrackAttributeBoxname,
            TrackidBox,
            plugin_data.axes.value,
            master_xml_path=plugin_data.master_xml_path.value,
            channel_seg_image=x_channel_seg,
            seg_image=x_seg,
            image=x,
            mask=x_mask,
            cluster_model=model_cluster,
            num_points=num_points,
            progress_bar=plugin.progress_bar,
            batch_size=plugin_data.batch_size.value,
        )
        nonlocal track_centroid_tree, track_centroid_list
        track_centroid_list = [
            k for k in _trackmate_objects.unique_track_centroid.keys()
        ]
        track_centroid_tree = spatial.cKDTree(track_centroid_list)
        _refreshStatPlotData()
        plugin_data.compute_button.enabled = False

    @change_handler(plugin_data.track_csv_path, init=False)
    def _track_csv_path_change(value):

        plugin_data.compute_button.enabled = True

    @change_handler(plugin_data.spot_csv_path, init=False)
    def _spot_csv_path_change(value):

        plugin_data.compute_button.enabled = True

    @change_handler(plugin_data.edges_csv_path, init=False)
    def _edges_csv_path_change(value):

        plugin_data.compute_button.enabled = True

    @change_handler(plugin_data.size_catagories, init=False)
    def _size_catagories_change(value):

        plugin_data.compute_button.enabled = True

    @change_handler(plugin_data.master_xml_path, init=False)
    def _master_xml_path_change(value):

        plugin_data.compute_button.enabled = True

    @change_handler(plugin_data.xml_path, init=False)
    def _xml_path_change(value):

        plugin_data.compute_button.enabled = True

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
        plugin_data.compute_button.enabled = True
        # Trigger model change
        selected.changed(selected.value)

    @change_handler(plugin.track_model_type, init=False)
    def _change_track_model_type(value):

        plugin.track_model_type.value = value
        select_track_nature()
        plot_main()
        show_phenotype()

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
        plugin_data.compute_button.enabled = True
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
        plugin_data.compute_button.enabled = True
        value = plugin_data.axes.value
        print(f"axes is {value}")

    return plugin
