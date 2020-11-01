import pandas as pd
from ast import literal_eval
import napari
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QWidget

LABEL_MAPPING = "./sentinel_to_zarr/class_map.txt"
RAW_PATH = "/media/draga/My Passport/Zarr/55HBU_Raw/10m_Res.zarr"
INTERPOLATED_PATH = "/media/draga/Elements/55HBU_GapFilled_Multiscale.zarr"
LABELS_PATH = "/media/draga/My Passport/55HBU_Multiscale_Labels.zarr"
LEVEL = 1

USED_LAYERS = ["FRE_B2", "FRE_B3", "FRE_B4", "FRE_B8","Raw_FRE_B2", "Raw_FRE_B3", "Raw_FRE_B4", "Raw_FRE_B8"]

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

    colour_dict = {
        0: (0, 0, 0, 0),
        1: (1.0, 0.39215686274509803, 0.39215686274509803),
        2: (0.0, 0.1568627450980392, 0.7843137254901961),
        3: (0.5803921568627451, 0.19607843137254902, 0.6470588235294118),
        4: (0.9921568627450981, 0.6823529411764706, 0.8980392156862745),
        5: (0.7764705882352941, 0.7686274509803922, 0.8470588235294118),
        6: (0.19607843137254902, 0.5882352941176471, 0.0),
        7: (1.0, 1.0, 0.7529411764705882),
        8: (0.8862745098039215, 0.9529411764705882, 0.6392156862745098),
        9: (0.5686274509803921, 0.5098039215686274, 0.4117647058823529),
        10: (1.0, 1.0, 0.0),
        11: (1.0, 0.8235294117647058, 0.49019607843137253),
        12: (1.0, 0.5490196078431373, 0.0),
        13: (0.7450980392156863, 0.9019607843137255, 0.35294117647058826),
    }

    with napari.gui_qt():
        napari.utils.dask_utils.resize_dask_cache(mem_fraction=0.1)
        viewer = napari.Viewer()
        viewer.open(
            RAW_PATH, 
            scale=(365 / 108, 1, 1, 1), 
            visible=False
        )
        
        for layer in viewer.layers:
            layer.name = f"Raw_{layer.name}"

        viewer.open(
            INTERPOLATED_PATH,
            scale=(365 / 73, 1, 1, 1),
            multiscale=True,
        )

        to_delete = []
        for i,layer in enumerate(viewer.layers):
            if not layer.name in USED_LAYERS:
                to_delete.append(i)
        to_delete.reverse()
        for i in to_delete:
            viewer.layers.pop(i)

        viewer.open(
            LABELS_PATH,
            scale=(365*2, 1, 1, 1),
            layer_type="labels",
            properties=label_properties,
            color=colour_dict,
            opacity=0.4,
        )

        # aribitrarily grab red layer
        intensity_plot_im = viewer.layers["FRE_B2"].data[LEVEL]
        # create the intensity plot
        with plt.style.context("dark_background"):
            intensity_canvas = FigureCanvas(Figure(figsize=(5, 3)))
            intensity_axes = intensity_canvas.figure.subplots()
            intensities = intensity_plot_im[:, 0, 0, 0]

            intensity_line = intensity_axes.plot(
                np.arange(intensity_plot_im.shape[0]), intensities
            )[
                0
            ]  # returns line list
            position_line = intensity_axes.axvline(x=0, c="C1")
            position_line.set_zorder(-1)  # keep the spectra in front
            minval, maxval = np.min(intensities), np.max(intensities)
            range_ = maxval - minval
            centre = (maxval + minval) / 2
            min_y = centre - 1.05 * range_ / 2
            max_y = centre + 1.05 * range_ / 2
            intensity_axes.set_ylim(min_y, max_y)
            intensity_axes.set_xlabel("time")
            intensity_axes.set_ylabel("intensity")
            title = intensity_axes.set_title("Intensity at: coord=(0, 0)")
            intensity_canvas.figure.tight_layout()

        toolbar = NavigationToolbar2QT(intensity_canvas, viewer.window._qt_window)
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(toolbar)
        layout.addWidget(intensity_canvas)
        viewer.window.add_dock_widget(widget)

        # create a function to update the vertical bar corresponding to timepoint
        def update_plot(axis_event):
            axis = axis_event.axis
            if axis != 1:
                return
            x = axis_event.value
            # only every (107/73) = 1.4657 steps do we see a new interpolated image
            if x % 3 == 0: 
                position_line.set_data([x/1.4657, x/1.4657], [0, 1])
                intens_str, coord_str = title.get_text().split(":")
                title.set_text(intens_str + ":" + coord_str)
                intensity_canvas.draw_idle()

        # connect the function to the dims axis
        viewer.dims.events.current_step.connect(update_plot)

        def update_intensity(layer, event):
            xs, ys = intensity_line.get_data()
            coords_full = tuple(np.round(layer.coordinates).astype(int) // 2 ** LEVEL)

            in_range = all(
                coords_full[i] in range(intensity_plot_im.shape[i])
                for i in range(intensity_plot_im.ndim)
            )
            coords = coords_full[-2:]  # rows, columns
            if in_range:
                print(f"Updating plot for {coords}...")
                new_ys = intensity_plot_im[:, :, coords[0], coords[1]]
                min_y = np.min(new_ys)
                max_y = np.max(new_ys)
                intensity_axes.set_ylim(min_y, max_y)
                intensity_line.set_data(xs, new_ys)
                intens_str, coords_str = title.get_text().split(":")
                title.set_text(intens_str + ": " + str(coords))
                intensity_canvas.draw_idle()

        for layer in viewer.layers:
            # add a click callback to each layer to update the spectrum being viewed
            layer.mouse_drag_callbacks.append(update_intensity)


def get_label_properties():
    df = pd.read_csv(LABEL_MAPPING)

    dicts = df.to_dict("split")
    classes = list(df["class"])
    colors = [tuple([v / 255 for v in literal_eval(val)]) for val in list(df["colour"])]

    label_properties = {"class": ["None"] + classes}

    colour_indices = [i for i in range(df.shape[0] + 1)]
    colours = [(0, 0, 0, 0)] + colors
    colour_dict = dict(zip(colour_indices, colours))

    return label_properties, colour_dict


main()
