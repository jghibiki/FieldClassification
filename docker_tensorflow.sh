sudo nvidia-docker run -it -v /home/jgoetze/FieldClassification:/mnt field-classification \
    python train.py --batch_size 5 --checkpoint_every 75 --num_epochs 3 --summary_every 1 --summary_train_dir summaries/nlcd-fixed-images-5-conv-layers-many-images-3x3-filter

