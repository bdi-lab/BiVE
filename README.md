# BiVE: Learning Representations of Bi-Level Knowledge Graphs for Reasoning beyond Link Prediction
This code is an implementation of the paper, "Learning Representations of Bi-Level Knowledge Graphs for Reasoning beyond Link Prediction (AAAI 23)".

This code is based on the [OpenKE](https://github.com/thunlp/OpenKE) implementation, which is an open toolkit for knowledge graph embedding.


When you use this code, please cite our paper.

## Usage

### Data Augmentation by Random Walks

Use `augment.py` to perform data augmentation.

```
python augment.py [data] [conf] [count]
```
- `[data]`: name of the dataset. The name should be the directory name of the dataset contained in the `./benchmarks` folder.
- `[conf]`: threshold of the confidence score, i.e., $\tau$ in the paper.

### BiVE

To train BiVE_QuatE, use `bive_quate.py`.

```
CUDA_VISIBLE_DEVICES=0 python bive_quate.py [data] [learning_rate] [regul_rate] [epoch] --meta [weight_high] --aug [weight_aug]
```

To train BiVE_BiQUE, use `bive_bique_add.py`.

```
CUDA_VISIBLE_DEVICES=0 python bive_bique_add.py [data] [learning_rate] [regul_rate] [epoch] --meta [weight_high] --aug [weight_aug]
```
