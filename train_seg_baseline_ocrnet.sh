#/bin/bash
python ./train_seg_baseline_transfomer.py \
 train_seg_baseline_swin.sh \
 0 \
 50 \
 path_to_data \
 VOC2012 \
 21 \
 4 \
 nonRW \
 1 \
 SegmentationClassAug \
 16 \
 1e-4 \
 5e-4 \
 0.9 \
 50 \
 0 \
 0 \
 None \
 None \
 None \
 False \
 None \
 None \
 train.txt
