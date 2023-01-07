import napari
from _widget import plugin_wrapper_track


def show_napari():
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(widget=plugin_wrapper_track())
    # viewer.window.add_plugin_dock_widget(
    #    "vollseg-napari-trackmate", "NapaTrackMater"
    # )
    napari.run()


if __name__ == "__main__":

    show_napari()
