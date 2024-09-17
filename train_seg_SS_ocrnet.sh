#/bin/bash
python ./train_seg_SS_transformer.py \
 train_seg_SS_ocrnet.sh \
 0 \
 50 \
 path_to_data \
 VOC2012 \
 60 \
 4 \
 RW \
 1 \
 pascal_scribble_2012 \
 16 \
 1e-5 \
 5e-4 \
 0.9 \
 50 \
 1 \
 1 \
 model_path like:/home/root/CC4S/runs/1/model_best.pth.tar \
 random \
 P \
 True \
 Laplace \
 spixe \
 train.txt \
 0 \
 129
