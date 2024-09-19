# CC4S: Encouraging Certainty and Consistency in Scribble-Supervised Semantic Segmentation
[Zhiyi Pan](https://github.com/panzhiyi), [Haochen Sun](https://github.com/sun1233217T), [Peng Jiang*](https://github.com/sdujump), Ge Li, Changhe Tu, Haibin Ling

The work is based on URSS (https://github.com/panzhiyi/URSS) and has been accepted by TPAMI.

### Benchmark Setup

##### Dataset

*scribble_shrink* and *scribble_drop* are available at [here](https://drive.google.com/drive/folders/1q2PvbQVOdIY9S-qjh85ohM66svzp9wnp).  The *scribble_sup* dataset can be downloaded on [jifengdai.org/downloads/scribble_sup/](https://jifengdai.org/downloads/scribble_sup/).

##### Environment

```
pip install -r requirements.txt
```

##### Checkpoints

You can download our [pretrained model](https://drive.google.com/drive/folders/1pA0OKI5dczI5rgk-tPaZrmmFwkbm37DE?usp=sharing) to reproduce the results reported in the paper.

### Pseudo-label Generation Phase

##### Baseline

Please modify the dataset file path in **train_seg_baseline.sh** and run:

```
sh train_seg_baseline.sh
```

##### First-stage training with Uncertainty Reduction on Neural Representation

Please modify the dataset file path in **train_seg_UR.sh** and run:

```
sh train_seg_UR.sh
```

the model will be saved in ./runs/ 

##### Second-stage training to refine the model with soft entropy loss

Please modify the model(obtained by first-stage training) file path in **train_seg_SS.sh** and run: 

```
sh train_seg_SS.sh
```

##### Evaluate

Please modify the model (obtained by second-stage training) file path and save path in **evaluate.sh** and run: 

```
sh evaluate.sh
```

All the computations are carried out on NVIDIA TITAN RTX GPUs.

### Pseudo-label Retraining Phase

##### Preparation for color constraint regularizer

Please modify the dataset file path and save the path in **/tool/scribblesup.m** and run in Matlab.  

##### Refinement
Do the same operation as stage1:  
  

Please modify the dataset file path in **train_seg_UR_stage2.sh** and run:

```
sh train_seg_UR_stage2.sh
```

Please modify the model(obtained by first-stage training) file path in **train_seg_SS_stage2.sh** and run: 

```
sh train_seg_SS_stage2.sh
```
Evaluate is the same as before.

### Citation

If you find our code or paper useful, please cite:

```
@article{pan2024cc4s,
  title={CC4S: Encouraging Certainty and Consistency in Scribble-Supervised Semantic Segmentation},
  author={Pan, Zhiyi and Sun, Haochen and Jiang, Peng and Li, Ge and Tu, Changhe and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```


