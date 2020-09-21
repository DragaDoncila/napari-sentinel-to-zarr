import numpy as np
from napari_sentinel_to_zarr.sentinel_to_zarr import to_ome_zarr
from napari_sentinel_zip.napari_sentinel_zip import reader_function


def test_read_write():
    zipfn = "/media/draga/My Passport/pepsL2A_zip_img/55HBU"
    data = reader_function(zipfn)
    outfn = "/media/draga/My Passport/WriterPlugin/55HBU.zarr"
    out_paths = to_ome_zarr(outfn, data)


# # tmp_path is a pytest fixture
# def test_reader(tmp_path):
#     """An example of how you might test your plugin."""

#     # write some fake data using your supported file format
#     my_test_file = str(tmp_path / "myfile.npy")
#     original_data = np.random.rand(20, 20)
#     np.save(my_test_file, original_data)

#     # try to read it back in
#     reader = napari_get_reader(my_test_file)
#     assert callable(reader)

#     # make sure we're delivering the right format
#     layer_data_list = reader(my_test_file)
#     assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
#     layer_data_tuple = layer_data_list[0]
#     assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

#     # make sure it's the same as it started
#     np.testing.assert_allclose(original_data, layer_data_tuple[0])


# def test_get_reader_pass():
#     reader = napari_get_reader("fake.file")
#     assert reader is None

if __name__ == '__main__':
    test_read_write()