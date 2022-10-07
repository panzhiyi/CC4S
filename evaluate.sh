#/bin/bash
nohup \
python ./evaluate.py \
 evaluate.sh \
 3 \
 50 \
 /home/ubuntu/JP/data/VOC2012/ \
 VOC2012 \
 21 \
 2 \
 RW \
 1 \
 /home/ubuntu/JP/projects/URSS-main/runs/iter0-self1-scribble2/model_best.pth.tar \
 /home/ubuntu/JP/data/VOC2012/output_path/ \
 False \
 pascal_2012_scribble \
 0
  