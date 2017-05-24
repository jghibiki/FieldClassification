# Field Classification Research


# Requirements

To prepare the image dataset and train the model you will need the following requirements:
- [Docker](https://www.docker.com/) - used for easily installing a reproducable environment by using container technology
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) - Allows docker containers to access the nvidia cuda API.
- [imageinfo](http://manpages.ubuntu.com/manpages/zesty/man1/imageinfo.1.html) - used for getting the dimensions of images while converting from .tif to .png


# Preparing Dataset
## Dataset Download instructions:
1. Navigate to [Earth Explorer](https://earthexplorer.usgs.gov/)
2. Create an account and log in. Account creation is required in order to download images above a certian size.
3. *TODO* finish download instructions

## Reprojecting ##
1. Copy all image zip files into unprocessed_images/ and cd into unprocessed_images
2. Unzip all files with ```unzip \*.(ZIP|zip)```
3. Run ```./generate_images.sh```. This will reproject the downloaded images.
4. Edit ```./generate_nlcd_layer.sh``` to reflect where you downloaded nlcd_2011_landcover_2011_edition_2014_10_10.img to.
5. Run ```./generate_nlcd_layer.sh``` This will use the rerojected images to cut and reproject a file containing label data for a corresponding region.
6. Ensure FieldClassification/images exists. If it does not create it wil ```mkdir ../images``` (this command is reletive to unprocessed_images/).
7. Run ```./copy_images.sh``` to copy the generated PNG files to ../images

## Cut images into swatches ##
1. Run ```mkdir raw_images```
2. ./docker_run.sh "python generate_dataset.py" 
3. ./docker_run.sh "python calc_splits.py"

