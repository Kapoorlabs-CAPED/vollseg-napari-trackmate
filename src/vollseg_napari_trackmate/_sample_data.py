"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from napatrackmater import test_tracks_xenopus

def get_test_tracks_xenopus():
    
    return [(test_tracks_xenopus(), {'name': 'test_tracks_xenopus'})]
