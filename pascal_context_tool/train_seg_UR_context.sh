#/bin/bash
python ./train_seg_UR_context.py \
 train_seg_UR_context.sh \
 3,2,1 \
 101 \
 /home/ubuntu/JP/data/VOC2012 \
 VOC2012 \
 60 \
 4 \
 RW \
 1 \
 iter1/res101-pascal_context_scribble-iter0-scribble1-scribble2-0.9+scr \
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
 Laplace_context \
 spixe_context

