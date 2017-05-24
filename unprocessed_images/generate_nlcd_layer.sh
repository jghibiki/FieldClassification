#!/bin/bash

# converts base nlcd to re-warped verstion, takes a while - comment out after the first time to save time
# !! Change path of nlcd_2011_landcover_2011_edition_2014_10_10.img to path of NLCD image. The NLCD image can be downloaded at https://www.mrlc.gov/nlcd11_data.php !!
gdalwarp -t_srs EPSG:4326 -overwrite ~/nlcd_2011_landcover_2011_edition_2014_10_10/nlcd_2011_landcover_2011_edition_2014_10_10.img ./corrected_nlcd_whole.tif

for d in ./*/; do
    cd "$d";
    echo "Processing $d";

    rm *nlcd_layer.tif

    # gets the georeferenced height and width of the image
    s=`gdalinfo m_*.tif | head -n 3 | tail -n 1 | sed 's/,//g' | awk -F ' ' '{print $3 " " $4}'`

    # crops/cuts, and reprojects label image to the same projection system and resolution 
    gdalwarp -q \
        -ts $s \
        -t_srs EPSG:4326 \
        -cutline *.shp \
        -crop_to_cutline -of GTiff ../corrected_nlcd_whole.tif corrected_nlcd_layer.tif

    # get height and width of PNG image.
    size=`imageinfo --geom image.png`

    # split string on x character
    sizearray=(${size//x/ })

    # export georeferenced image to raw png with height and width of the data image.
    gdal_translate -of png -outsize ${sizearray[0]} ${sizearray[1]} corrected_nlcd_layer.tif label.png

    cd ..

done;

