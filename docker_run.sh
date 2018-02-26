if [ -z "$1" ]
  then
      echo "No argument supplied"
fi
nvidia-docker run -it -v `pwd`:/mnt field-classification $1
