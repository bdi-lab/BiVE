# BiVE: Learning Representations of Bi-Level Knowledge Graphs for Reasoning beyond Link Prediction
This code is an implementation of the paper, "Learning Representations of Bi-Level Knowledge Graphs for Reasoning beyond Link Prediction (AAAI 23)".

This code is based on the [OpenKE](https://github.com/thunlp/OpenKE) implementation, which is an open toolkit for knowledge graph embedding. Additional codes are written by Chanyoung Chung.

When you use this code, please cite our paper.

```
Chanyoung Chung and Joyce Jiyoung Whang, Learning Representations of Bi-Level Knowledge Graphs for Reasoning beyond Link Prediction, AAAI 23
```

## Usage

### Data Augmentation by Random Walks

Use `augment.py` to perform data augmentation.

```
python augment.py [data] [conf]
```
- `[data]`: name of the dataset. The name should be the directory name of the dataset contained in the `./benchmarks` folder.
- `[conf]`: threshold of the confidence score, i.e., $\tau$ in the paper.

### BiVE

To train BiVE-Q, use `bive_q.py`.

```
CUDA_VISIBLE_DEVICES=0 python bive_q.py [data] [learning_rate] [regul_rate] [epoch] --meta [weight_high] --aug [weight_aug] --lp/tp/clp
```

To train BiVE-B, use `bive_b.py`.

```
CUDA_VISIBLE_DEVICES=0 python bive_b.py [data] [learning_rate] [regul_rate] [epoch] --meta [weight_high] --aug [weight_aug] --lp/tp/clp
```
