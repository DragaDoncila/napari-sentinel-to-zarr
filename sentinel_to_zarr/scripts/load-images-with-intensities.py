import pandas as pd
from ast import literal_eval
import napari
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np

LABEL_MAPPING = "./sentinel_to_zarr/class_map.txt"
RAW_PATH = "/media/draga/My Passport/Zarr/55HBU_Raw/10m_Res.zarr"
INTERPOLATED_PATH = "/media/draga/My Passport/55HBU_GapFilled_Multiscale.zarr"
LABELS_PATH = "/media/draga/My Passport/Zarr/55HBU_Multiscale_Labels.zarr/0/"

def main():
    label_properties, colour_dict = get_label_properties()

    with napari.gui_qt():
        napari.utils.dask_utils.resize_dask_cache(mem_fraction=0.1)
        viewer = napari.Viewer()
        viewer.open(
            RAW_PATH,
            scale=(365/108, 1, 1, 1),
            visible=False
            )
        for layer in viewer.layers:
            layer.name = f"Raw_{layer.name}"

        viewer.open(
            INTERPOLATED_PATH,
            scale=(365/73, 1, 1, 1), 
            multiscale=True,
        )

        viewer.open(
            LABELS_PATH,
            scale=(1,1),
            layer_type="labels",
            properties=label_properties,
            color=colour_dict,
            opacity=0.4
        )

        # aribitrarily grab red layer
        intensity_plot_im = viewer.layers['FRE_B2'].data[0]
        # create the intensity plot
        with plt.style.context('dark_background'):
            intensity_canvas = FigureCanvas(Figure(figsize=(5, 3)))
            intensity_axes = intensity_canvas.figure.subplots()
            intensities = intensity_plot_im[:, 0, 0, 0]

            intensity_line = intensity_axes.plot(np.arange(intensity_plot_im.shape[0]), intensities)[0]  # returns line list
            position_line = intensity_axes.axvline(x=0, c='C1')
            position_line.set_zorder(-1)  # keep the spectra in front
            minval, maxval = np.min(intensities), np.max(intensities)
            range_ = maxval - minval
            centre = (maxval + minval) / 2
            min_y = centre - 1.05 * range_ / 2
            max_y = centre + 1.05 * range_ / 2
            intensity_axes.set_ylim(min_y, max_y)
            intensity_axes.set_xlabel('time')
            intensity_axes.set_ylabel('intensity')
            title = intensity_axes.set_title('Intensity at: coord=(0, 0)')
            intensity_canvas.figure.tight_layout()


        # add the plot to the viewer
        viewer.window.add_dock_widget(intensity_canvas)


        # create a function to update the plot
        def update_plot(axis_event):
            axis = axis_event.axis
            if axis != 0:
                return
            x = axis_event.value
            position_line.set_data([x, x], [0, 1])
            intens_str, coord_str = title.get_text().split(':')
            title.set_text(intens_str + ":" + coord_str)
            intensity_canvas.draw_idle()


        # connect the function to the dims axis
        viewer.dims.events.axis.connect(update_plot)

        def update_intensity(layer, event):
            xs, ys = intensity_line.get_data()
            coords_full = tuple(np.round(layer._transforms.simplified(layer.coordinates)).astype(int))
            print(f"Updating plot for {coords_full[-2::]}...")
            if layer.name == 'Labels':
                in_range = all(coords_full[i] in range(intensity_plot_im.shape[i+2]) for i in range(2))
                coords = tuple((0, *coords_full))   # z, rows, columns
            else:
                in_range = all(coords_full[i] in range(intensity_plot_im.shape[i])
                    for i in range(intensity_plot_im.ndim))
                coords = coords_full[1:]  # z, rows, columns
            if in_range:
                new_ys = intensity_plot_im[:, coords[0], coords[1], coords[2]]
                min_y = np.min(new_ys)
                max_y = np.max(new_ys)
                intensity_axes.set_ylim(min_y, max_y)
                intensity_line.set_data(xs, new_ys)
                intens_str, coords_str = title.get_text().split(':')
                title.set_text(intens_str + ': ' + str(coords))
                intensity_canvas.draw_idle()

        for layer in viewer.layers:
            # add a click callback to each layer to update the spectrum being viewed
            layer.mouse_drag_callbacks.append(update_intensity)

def get_label_properties():
    df = pd.read_csv(LABEL_MAPPING)

    dicts = df.to_dict('split')
    classes = list(df['class'])
    colors = [tuple([v / 255 for v in literal_eval(val)]) for val in list(df['colour'])]

    label_properties = {
        'class': ['None'] + classes
    }

    colour_indices = [i for i in range(df.shape[0] + 1)]
    colours = [(0, 0, 0, 0)] + colors
    colour_dict = dict(zip(colour_indices, colours))

    return label_properties, colour_dict

main()
