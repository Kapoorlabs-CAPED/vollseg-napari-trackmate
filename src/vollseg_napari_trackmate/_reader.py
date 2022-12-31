"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers.
"""
import codecs
import os
import xml.etree.ElementTree as et

import numpy as np
from napatrackmater.bTrackmate import load_tracks


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str

    Returns
    -------
    function
    """

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):

    if os.path.isfile(path):

        root = et.fromstring(codecs.open(path, "r", "utf8").read())

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

    add_kwargs = {}

    layer_type = "tracks"

    return [(np.asarray(tracklets), add_kwargs, layer_type)]
