#/bin/bash
python ./train_seg_UR.py \
 train_seg_UR.sh \
 0 \
 50 \
 data_path like:/home/root/data/VOC2012 \
 VOC2012 \
 21 \
 4 \
 RW \
 1 \
 scribble_path in dataset like:pascal_scribble_2012 \
 16 \
 1e-4 \
 5e-4 \
 0.9 \
 50 \
 1 \
 1 \
 None \
 None \
 None \
 True \
 Laplace \
 spiex \
 train.txt

