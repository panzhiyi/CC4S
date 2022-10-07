#/bin/bash
python ./train_seg_SS.py \
 train_seg_SS.sh \
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
 spiex \
 train.txt
