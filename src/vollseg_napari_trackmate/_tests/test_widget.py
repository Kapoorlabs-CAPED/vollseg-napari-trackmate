import numpy as np

from vollseg_napari_trackmate import plugin_wrapper_track


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_plugin_wrapper_track(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = plugin_wrapper_track
    viewer.window.add_dock_widget(my_widget)
