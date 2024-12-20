#/bin/bash
python ./evaluate.py \
 evaluate.sh \
 0 \
 50 \
 data_path like:/home/root/data/VOC2012 \
 VOC2012 \
 21 \
 2 \
 RW \
 1 \
 checkpoint_path like:/home/root/URSS-main/runs/iter0-self1-scribble2/model_best.pth.tar \
 output_path like:/home/root/data/VOC2012/output_path/ \
 False \
 pascal_2012_scribble \
 1
  
