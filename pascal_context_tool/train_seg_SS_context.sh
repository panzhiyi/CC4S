#/bin/bash
python ./train_seg_SS_context.py \
 train_seg_SS_context.sh \
 0,1,2,3 \
 101 \
 /home/ubuntu/JP/data/VOC2012 \
 VOC2012 \
 60 \
 4 \
 RW \
 1 \
 pascal_context_scribble  \
 14 \
 1e-5 \
 5e-4 \
 0.9 \
 50 \
 1 \
 1 \
 /home/ubuntu/JP/projects/URSS-main/runs/res101-pascal_context_scribble-iter0-scribble1/model_best.pth.tar \
 random \
 P \
 True \
 Laplace_context \
 spixe_context
