# BiVE: Learning Representations of Bi-level Knowledge Graphs for Reasoning beyond Link Prediction
This code is an implementation of the following [paper](https://arxiv.org/abs/2302.02601):

> Chanyoung Chung and Joyce Jiyoung Whang, Learning Representations of Bi-level Knowledge Graphs for Reasoning beyond Link Prediction, AAAI Conference on Artificial Intelligence (AAAI), 2023.

This code is based on the [OpenKE](https://github.com/thunlp/OpenKE) implementation, which is an open toolkit for knowledge graph embedding. Additional codes are written by Chanyoung Chung (chanyoung.chung@kaist.ac.kr).

When you use this code or data, please cite our paper.

```bibtex
@inproceedings{bive,
	author={Chanyoung Chung and Joyce Jiyoung Whang},
	title={Learning Representations of Bi-level Knowledge Graphs for Reasoning beyond Link Prediction},
	booktitle={Proceedings of the 37th AAAI Conference on Artificial Intelligence},
	year={2023},
	pages={4208--4216},
	doi={10.1609/aaai.v37i4.25538}
}
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

To train BiVE-Q, use `bive_q_new.py`.

```
CUDA_VISIBLE_DEVICES=0 python bive_q_new.py [data] [learning_rate] [regul_rate] [epoch] --meta [weight_high] --aug [weight_aug] --lp/tp/clp
```

To train BiVE-B, use `bive_b_new.py`.

```
CUDA_VISIBLE_DEVICES=0 python bive_b_new.py [data] [learning_rate] [regul_rate] [epoch] --meta [weight_high] --aug [weight_aug] --lp/tp/clp
```
