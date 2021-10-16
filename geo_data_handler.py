# -*- coding: utf-8 -*-

from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly

def geotiff_to_array(in_fp):
    """ Convert a GeoTiff into an array. May handle different formats, 
        but array_to_raster() below only handles GeoTiff. 
    
    Args:
        :in_fp:     input filepath to GeoTiff Raster
    
    Returns:
        Numpy Array
    
    Raises:
        AttributeError if input is not a recognized raster format
    """

    gdal_ras = gdal.Open(in_fp, GA_ReadOnly)   
    cols = gdal_ras.RasterXSize
    rows = gdal_ras.RasterYSize  
    array = gdal_ras.ReadAsArray(0, 0, cols, rows).astype(int)
                
    return array

def array_to_raster(inTiff_fp, array, out_fp, dataType=gdal.GDT_Float32):    
    """
    Save a raster from a C order array. Standard output is GeoTiff.
    The attributes of an exisiting raster are used for the new output raster    
    
    Changed after the original
    http://gis.stackexchange.com/questions/58517
    /python-gdal-save-array-as-raster-with-projection-from-other-file
    
    Args:
        :inTiff_fp: filepath to an exisiting GeoTiff file, the attributes from this file are used
                       to create the new one
        :array:     is the array that will be saved as a GeoTiff   
        :out_fp:    is the filepath of the desired output tiff
    
    Returns:
        GDAL dataset and a rasterband. Both can be discarded, as function saves file
        to disk before returning both.
    
    Raises:
        
    """
    
    inDataset = gdal.Open(inTiff_fp, GA_ReadOnly)

    # You need to get those values like you did.
    x_pixels = inDataset.RasterXSize  # number of pixels in x
    y_pixels = inDataset.RasterYSize  # number of pixels in y
    PIXEL_SIZE = inDataset.GetGeoTransform()[1]   # size of the pixel...        
    x_min = inDataset.GetGeoTransform()[0] 
    y_max = inDataset.GetGeoTransform()[3]   # x_min & y_max are like the "top left" corner.
    wkt_projection = inDataset.GetProjectionRef()

    driver = gdal.GetDriverByName('GTiff')

    outDataset = driver.Create(
        out_fp,
        x_pixels,
        y_pixels,
        1,
        dataType, 
        options = ['COMPRESS=LZW'])

    outDataset.SetGeoTransform((
        x_min,      # 0
        PIXEL_SIZE, # 1
        0,          # 2
        y_max,      # 3
        0,          # 4
        -PIXEL_SIZE))

    outDataset.SetProjection(wkt_projection)
    outDataset.GetRasterBand(1).WriteArray(array)
    outDataset.FlushCache()  # Write to disk.
    
    #If you need to return, remenber to return also the dataset because the band don't live without dataset
    return outDataset, outDataset.GetRasterBand(1)  