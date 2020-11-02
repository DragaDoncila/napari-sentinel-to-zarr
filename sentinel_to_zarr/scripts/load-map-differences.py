import napari
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import dask.array as da

label_properties = {
    "class": [
        "None",
        "Urban",
        "Water",
        "Deciduous Woody Horticulture",
        "Evergreen Woody Horticulture",
        "Non-Woody Horticulture",
        "Native Woody Cover",
        "Hardwood Plantation",
        "Softwood Plantation",
        "Bare Ground",
        "Brassica",
        "Cereals",
        "Legumes",
        "Grassland",
    ]
}

MAP_1_PATH = "/media/draga/My Passport/Map_Differences/55HBU_Map_1.zarr"
MAP_2_PATH = "/media/draga/Elements/Map_Differences/55HBU_Map_4.zarr"
MAX_LAYER = 3

def main():
    map_1_layers = []
    map_2_layers = []
    for i in range(MAX_LAYER + 1):
        map_1_layer = da.from_zarr(MAP_1_PATH + f"/{i}")
        map_2_layer = da.from_zarr(MAP_2_PATH + f"/{i}")

        map_1_layers.append(map_1_layer)
        map_2_layers.append(map_2_layer)

    difference_layers = []
    for map_1_layer, map_2_layer in zip(map_1_layers, map_2_layers):
        difference = map_2_layer - map_1_layer
        difference_binary = np.where(difference != 0, 1, 0)
        difference_layers.append(difference_binary)

    difference_colors = {
        1: "white"
    }


    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.open(
            MAP_1_PATH,
            properties=label_properties,
            visible=True
        )
        viewer.open(
            MAP_2_PATH,
            properties=label_properties,
            visible=False
        )
        viewer.add_labels(
            difference_layers,
            name="Difference",
            color=difference_colors
        )





main()