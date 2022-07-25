from typing import Tuple
from cv2 import transform
from osgeo import gdal
from albumentations import Compose, Normalize, RandomCrop

def write_geotiff(output_tif, ncols, nrows,
                  xmin, xres,ymax, yres,
                 raster_srs, label_arr):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, ncols, nrows, len(label_arr), gdal.GDT_Byte)
    out_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    out_ds.SetProjection(raster_srs.ExportToWkt())
    for i in range(len(label_arr)):
        outband = out_ds.GetRasterBand(i+1)
        outband.WriteArray(label_arr[i])
        #outband.SetNoDataValue(0)
        outband.FlushCache()
    out_ds = None

def get_transforms(crop=(512, 512), normalize=True) -> Tuple[list, list]:
    train_transforms = list()
    validation_tranforms = list()
    if crop is not None:
        train_transforms.append(RandomCrop(*crop, always_apply=True))
    if normalize:
        train_transforms.append(Normalize())
        validation_tranforms.append(Normalize())
    return train_transforms, validation_tranforms

