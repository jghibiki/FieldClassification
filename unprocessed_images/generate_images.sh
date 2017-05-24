
for d in ./*/; do
    cd "$d";
    echo "Processing $d";

    rm -f corrected_image.tif ;
    
    # reprojects data image to EPSG:4326
    gdalwarp -q -t_srs EPSG:4326  -overwrite m_*.tif corrected_image_layer.tif

    # convert data image to PNG
    gdal_translate -of png corrected_image_layer.tif image.png
    
    cd ..;
done;
