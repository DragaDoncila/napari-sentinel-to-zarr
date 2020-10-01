"""
Viewing pixel intensity imaging data in napari.

Adapted from Juan Nunez-Iglesias' script for viewing mass spectrometry data in napari
"""

import numpy as np
import napari
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import tifffile


# create qt application context
with napari.gui_qt():
    # create the viewer and open image
    viewer = napari.Viewer()
    viewer.open("/media/draga/My Passport/Zarr/55HBU_Multiscale_Zarr_SingleChannel.zarr")
    im = viewer.layers[0].data[0]

    # create the intensity plot
    with plt.style.context('dark_background'):
        intensity_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        intensity_axes = intensity_canvas.figure.subplots()
        intensities = im[:, 0, 0, 0]

        intensity_line = intensity_axes.plot(np.arange(im.shape[0]), intensities)[0]  # returns line list
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


    # grab the image layer
    layer = viewer.layers[0]


    # add a click callback to the layer to update the spectrum being viewed
    @layer.mouse_drag_callbacks.append
    def update_intensity(layer, event):
        xs, ys = intensity_line.get_data()
        coords_full = tuple(np.round(layer._transforms.simplified(layer.coordinates)).astype(int))
        print(coords_full)
        if all(coords_full[i] in range(im.shape[i])
                for i in range(im.ndim)):
            coords = coords_full[1:]  # rows, columns
            new_ys = im[:, coords[0], coords[1], coords[2]]
            min_y = np.min(new_ys)
            max_y = np.max(new_ys)
            intensity_axes.set_ylim(min_y, max_y)
            intensity_line.set_data(xs, new_ys)
            intens_str, coords_str = title.get_text().split(':')
            title.set_text(intens_str + ': ' + str(coords))
            intensity_canvas.draw_idle()