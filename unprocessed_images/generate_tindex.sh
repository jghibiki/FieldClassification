#!/bin/bash

# used for generating tile indexes for use with https://nassgeodata.gmu.edu/CropScape/
# the resulting zip files can be uploaded to the website as an area of interest in order to fetch a Crop Data Layer image corresponding to the source data image.

for d in ./*/; do
    cd "$d";
    gdaltindex out.shp *.tif;
    zip out.zip out.*;
    cd ..

done;
