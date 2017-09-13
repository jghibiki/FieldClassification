sudo nvidia-docker run -it -v /home/jgoetze/FieldClassification:/mnt field-classification \
    python train.py --batch_size 5 --checkpoint_every 500 --num_epochs 8 --summary_every 50 --summary_train_dir summaries/nlcd-fixed-images-5-conv-layers-many-images-3x3-filter-long-run

