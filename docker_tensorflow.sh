sudo docker run  --runtime=nvidia -it -v `pwd`:/mnt field-classification \
    python train.py --batch_size 5 --checkpoint_every 500 --num_epochs 8 --summary_every 50 --summary_train_dir summaries/aerial_dataset

