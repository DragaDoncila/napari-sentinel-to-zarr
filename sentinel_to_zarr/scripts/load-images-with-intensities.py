import napari
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import dask.array as da

RAW_PATH = "/media/draga/My Passport/Zarr/55HBU_Raw/10m_Res.zarr"
INTERPOLATED_PATH = "/media/draga/Elements/55HBU_GapFilled_Multiscale.zarr"

LABELS_PATH = "/media/draga/My Passport/55HBU_Multiscale_Labels.zarr"
ALTERNATE_LABELS_PATH = "/media/draga/Elements/Map_Differences/55HBU_Alternate_Map.zarr"
LEVEL = 1

USED_LAYERS = [
    "FRE_B2",
    "FRE_B3",
    "FRE_B4",
    "FRE_B8",
    "Raw_FRE_B2",
    "Raw_FRE_B3",
    "Raw_FRE_B4",
    "Raw_FRE_B8",
]


def main():
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

    with napari.gui_qt():
        napari.utils.dask_utils.resize_dask_cache(mem_fraction=0.1)

        # load raw images and prepend Raw onto their name
        viewer = napari.Viewer()
        viewer.open(
            RAW_PATH, 
            scale=(365 / 108, 1, 1, 1), 
            visible=False
        )
        for layer in viewer.layers:
            layer.name = f"Raw_{layer.name}"

        # load interpolated images 
        viewer.open(
            INTERPOLATED_PATH,
            scale=(365 / 73, 1, 1, 1),
            multiscale=True,
        )

        # just keep the layers in the list to mitigate clutter
        remove_unused_layers(viewer)

        # compute difference between two labels layers and display
        label_difference = get_label_difference()
        difference_colors = {
            1: "white"  # still not great contrast...
        }
        viewer.add_labels(
            label_difference,
            name="Label Difference",
            scale=(365 * 2, 1, 1, 1),
            color=difference_colors,
            visible=False
        )

        # add the two label images
        viewer.open(
            ALTERNATE_LABELS_PATH,
            name="Alternate Labels",
            scale=(365 * 2, 1, 1, 1),
            layer_type="labels",
            properties=label_properties,
            opacity=0.4,
            visible=False
        )
        viewer.open(
            LABELS_PATH,
            name= "Labels",
            scale=(365 * 2, 1, 1, 1),
            layer_type="labels",
            properties=label_properties,
            opacity=0.4,
            visible=False
        )

        # add NDVI layer
        NIR = viewer.layers["FRE_B8"].data
        red = viewer.layers["FRE_B4"].data
        ndvi_layer = get_ndvi_layer(NIR, red)
        viewer.add_image(
            ndvi_layer,
            name= "NDVI",
            scale=(365 / 73, 1, 1, 1),
            multiscale=True,
            contrast_limits=(0,1),
            colormap="RdYlGn",
            visible=False
        )

        NIR = NIR[LEVEL]
        red = red[LEVEL]
        # create the NDVI plot
        with plt.style.context("dark_background"):
            ndvi_canvas = FigureCanvas(Figure(figsize=(5, 3)))
            ndvi_axes = ndvi_canvas.figure.subplots()
            ndvi = get_ndvi(NIR, red, 0, 0)

            ndvi_line = ndvi_axes.plot(
                np.arange(NIR.shape[0]), ndvi
            )[
                0
            ]  # returns line list
            position_line = ndvi_axes.axvline(x=0, c="C1")
            position_line.set_zorder(-1)  # keep the time point in front
            # set y limits
            minval, maxval = np.min(ndvi), np.max(ndvi)
            range_ = maxval - minval
            centre = (maxval + minval) / 2
            min_y = centre - 1.05 * range_ / 2
            max_y = centre + 1.05 * range_ / 2
            ndvi_axes.set_ylim(min_y, max_y)
            ndvi_axes.set_xlabel("time")
            ndvi_axes.set_ylabel("NDVI")
            title = ndvi_axes.set_title("NDVI at: coord=(0, 0)")
            ndvi_canvas.figure.tight_layout()

        # add matplotlib toolbar
        toolbar = NavigationToolbar2QT(ndvi_canvas, viewer.window._qt_window)
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(toolbar)
        layout.addWidget(ndvi_canvas)
        viewer.window.add_dock_widget(widget)

        # create a function to update the vertical bar corresponding to timepoint
        def update_timepoint(axis_event):
            axis = axis_event.axis
            if axis != 1:
                return
            x = axis_event.value
            # only every (107/73) = 1.4657 steps do we see a new interpolated image
            if x % 3 == 0:
                position_line.set_data([x / 1.4657, x / 1.4657], [0, 1])
                intens_str, coord_str = title.get_text().split(":")
                title.set_text(intens_str + ":" + coord_str)
                ndvi_canvas.draw_idle()

        # connect the function to the dims axis
        viewer.dims.events.current_step.connect(update_timepoint)

        def update_ndvi(layer, event):
            xs, ys = ndvi_line.get_data()
            coords = np.round(layer.coordinates).astype(int)
            coords_display = tuple(coords)[-2:]
            coords_level = tuple(coords // 2 ** LEVEL)

            in_range = all(
                coords_level[i] in range(NIR.shape[i])
                for i in range(NIR.ndim)
            )
            if in_range:
                print(f"Updating plot for {coords_display}...")
                new_ys = get_ndvi(NIR, red, coords_level[-2], coords_level[-1])
                # set y limits
                minval, maxval = np.min(new_ys), np.max(new_ys)
                range_ = maxval - minval
                centre = (maxval + minval) / 2
                min_y = centre - 1.05 * range_ / 2
                max_y = centre + 1.05 * range_ / 2
                ndvi_axes.set_ylim(min_y, max_y)
                ndvi_line.set_data(xs, new_ys)
                intens_str, coords_str = title.get_text().split(":")
                title.set_text(intens_str + ": " + str(coords_display))
                ndvi_canvas.draw_idle()

        for layer in viewer.layers:
            # add a click callback to each layer to update the pixel being plotted
            layer.mouse_drag_callbacks.append(update_ndvi)

def remove_unused_layers(viewer):
    """Delete any layers not in the USED_LAYERS list"""
    to_delete = []
    for i, layer in enumerate(viewer.layers):
        if not layer.name in USED_LAYERS:
            to_delete.append(i)
    to_delete.reverse()
    for i in to_delete:
        viewer.layers.pop(i)

def get_ndvi(NIR, red, y, x):
    """Get NDVI of a particular pixel"""
    nir_intensities = NIR[:, 0, y, x].astype(np.float32)
    red_intensities = red[:, 0, y, x].astype(np.float32)

    intensity_sum = (nir_intensities + red_intensities)
    intensity_diff = (nir_intensities - red_intensities)

    ndvi =  da.divide(intensity_diff,intensity_sum)
    ndvi[da.isnan(ndvi)] = 0

    return ndvi

def get_ndvi_layer(NIR, red):
    """Get multiscale NDVI layer for display"""
    ndvi_levels = []
    for i in range(len(NIR)):
        current_nir = NIR[i].astype(np.float32)
        current_red = red[i].astype(np.float32)
        intensity_sum = current_nir + current_red
        intensity_diff = current_nir - current_red
        ndvi_layer = da.divide(intensity_diff, intensity_sum)
        ndvi_layer = da.nan_to_num(ndvi_layer)
        ndvi_levels.append(ndvi_layer)
    return ndvi_levels

def get_label_difference():
    """Get multiscale label difference between two maps"""
    map_1_layers = []
    map_2_layers = []
    for i in range(4):
        map_1_layer = da.from_zarr(LABELS_PATH + f"/{i}")
        map_2_layer = da.from_zarr(ALTERNATE_LABELS_PATH + f"/{i}")

        map_1_layers.append(map_1_layer)
        map_2_layers.append(map_2_layer)

    difference_layers = []
    for map_1_layer, map_2_layer in zip(map_1_layers, map_2_layers):
        difference = map_2_layer - map_1_layer
        difference_binary = np.where(difference != 0, 1, 0)
        difference_layers.append(difference_binary)

    return difference_layers   

main()
