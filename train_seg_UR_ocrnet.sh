#/bin/bash
python train_seg_UR_transformer.py \
 train_seg_UR_ocrnet.sh \
 0 \
 50 \
 path_to_data \
 VOC2012 \
 21 \
 6 \
 RW \
 1 \
 pascal_scribble_2012 \
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
 False \
 Laplace \
 spiex \
 train.txt

