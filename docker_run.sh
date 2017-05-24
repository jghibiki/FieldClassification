if [ -z "$1" ]
  then
      echo "No argument supplied"
fi
nvidia-docker run -it -v /home/jgoetze/FieldClassification:/mnt field-classification $1
