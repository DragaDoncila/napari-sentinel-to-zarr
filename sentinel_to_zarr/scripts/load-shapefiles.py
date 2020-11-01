import napari
from shapefile import Reader
import pycrs

PATH = "/home/draga/Honours/55HBU_RefData/55HBU_refdata"
PROJ_PATH = "/home/draga/Honours/55HBU_RefData/55HBU_refdata.prj"

sf = Reader(PATH)
shapes = sf.shapes()
first_shape = shapes[0]

shapes = [shape.points for shape in shapes]

crs = pycrs.load.from_file(PROJ_PATH)

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_shapes(shapes, shape_type = "polygon")