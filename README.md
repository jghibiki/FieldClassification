# Field Classification Research


# 0. Requirements

To prepare the image dataset and train the model you will need the following requirements:
- [Docker](https://www.docker.com/) - used for easily installing a reproducable environment by using container technology
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) - Allows docker containers to access the nvidia cuda API.
- [imageinfo](http://manpages.ubuntu.com/manpages/zesty/man1/imageinfo.1.html) - used for getting the dimensions of images while converting from .tif to .png


# 1. Preparing Dataset
## Dataset Download instructions:
1. Navigate to [Earth Explorer](https://earthexplorer.usgs.gov/)
2. Create an account and log in. Account creation is required in order to download images above a certian size.
3. *TODO* finish download instructions

## 2. Reprojecting ##
1. Copy all image zip files into unprocessed_images/ and cd into unprocessed_images
2. Unzip all files with ```unzip \*.(ZIP|zip)```
3. Run ```./generate_images.sh```. This will reproject the downloaded images.
4. Edit ```./generate_nlcd_layer.sh``` to reflect where you downloaded nlcd_2011_landcover_2011_edition_2014_10_10.img to.
5. Run ```./generate_nlcd_layer.sh``` This will use the rerojected images to cut and reproject a file containing label data for a corresponding region.
6. Ensure FieldClassification/images exists. If it does not create it wil ```mkdir ../images``` (this command is reletive to unprocessed_images/).
7. Run ```./copy_images.sh``` to copy the generated PNG files to ../images

## 3. Cut images into swatches ##
1. Run ```mkdir raw_images```
2. ```./docker_run.sh "python generate_dataset.py"```

## 4. Calculate Test/Train Datasets ##
Calculates test/train datasets and stores their corresponding id's in a .npy file (numpy file). This makes it easy to run multiple versions of the model on the same test and training sets.
1. ```./docker_run.sh "python calc_splits.py"```

## 5. Build the Docker Container ##
This step should only need to be done once initially, and then periodically to pull updated versions of the Tensorflow container.
1. ```./docker_build.sh``` - this may take a while.

## 6. Run the Model ##
1. Edit the second line of ```./docker_tensorflow.sh``` to set training parameters. 
2. Run ```./docker_tensorflow.sh``` - This will take a very long while as the model runs. While running, move on to the next step in a new terminal to view graphs.

## 7. View Model Progress/Statistics ##
1. Edit the absolute path to ```FieldClassification/``` on the first line of ```./docker_tensorboard.sh```
2. Run ```./docker_tensorboard.sh [path to summary dir]``` By default the summary dir is located in ```FieldClassification/output/summaries```. To view one run instead of all runs of the model pass the path to the specific model summary dir, e.g. ```output/summaries/my_model_name_here/```.

