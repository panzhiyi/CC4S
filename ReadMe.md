# CC4S: Encouraging Certainty and Consistency in Scribble-Supervised Semantic Segmentation
[Zhiyi Pan](https://github.com/panzhiyi), [Haochen Sun](https://github.com/sun1233217T), [Peng Jiang*](https://github.com/sdujump), Yunhai Wang, Changhe Tu, Anthony G. Cohn

The workis based on URSS(https://github.com/panzhiyi/URSS), the paper have been send to PAMI and is under review.

##### dataset

*scribble_shrink* and *scribble_drop* are available at [here](https://drive.google.com/drive/folders/1q2PvbQVOdIY9S-qjh85ohM66svzp9wnp).  The *scribble_sup* dataset can be downloaded on [jifengdai.org/downloads/scribble_sup/](https://jifengdai.org/downloads/scribble_sup/).

##### environment

```
pip install -r requirements.txt
```

##### baseline

Please modify the dataset file path in **train_seg_baseline.sh** and run:

```
sh train_seg_baseline.sh
```

##### first-stage training with Uncertainty Reduction on Neural Representation

Please modify the dataset file path in **train_seg_UR.sh** and run:

```
sh train_seg_UR.sh
```

the model will be saved in ./runs/ 

##### second-stage training to refine the model with soft self supervision loss

Please modify the model(obtained by first-stage training) file path in **train_seg_SS.sh** and run: 

```
sh train_seg_SS.sh
```

##### evaluate

Please modify the model(obtained by second-stage training) file path and save path in **evaluate.sh** and run: 

```
sh evaluate.sh
```

All the computations are carried out on NVIDIA TITAN RTX GPUs.

### Refine stage:

##### preparation for color constraint regularizer

Please modify the dataset file path and save path in **/tool/scribblesup.m** and run in matlab.  

##### refinement
Do the operation like stage1:  
  

Please modify the dataset file path in **train_seg_UR_stage2.sh** and run:

```
sh train_seg_UR_stage2.sh
```

Please modify the model(obtained by first-stage training) file path in **train_seg_SS_stage2.sh** and run: 

```
sh train_seg_SS_stage2.sh
```
Evaluate is the same as before.





