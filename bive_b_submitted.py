import openke
from openke.config import Trainer, Tester
from openke.module.model import BiVE_B
from openke.module.loss import SoftplusLoss_submitted
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse
import os

parser = argparse.ArgumentParser(description="Hyperparameters")
parser.add_argument('data')
parser.add_argument('alpha', type=float, help="learning rate")
parser.add_argument('regul', type=float, help="regul_rate")
parser.add_argument('epoch', type=int, help="epoch")
parser.add_argument('--meta', type=float, default=0.0, help="weight_meta")
parser.add_argument('--aug', type=float, default=0.0, help="weight_aug")
parser.add_argument('--lp', default=False, action="store_true")
parser.add_argument('--tp', default=False, action="store_true")
parser.add_argument('--clp', default=False, action="store_true")
parser.add_argument('--serial', type=int, default=0)
args = parser.parse_args()

### SET RANDOM SEED ###
import torch
import numpy as np
import random
random.seed(args.serial)
np.random.seed(args.serial)
torch.manual_seed(args.serial)

OMP_NUM_THREADS=8
torch.set_num_threads(8)

### PRINT INFO ###
print("data: {} alpha: {} regul: {} meta: {} aug: {} epoch: {}".format(args.data, args.alpha, args.regul, args.meta, args.aug, args.epoch))

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path="./benchmarks/{}/base/".format(args.data), 
	nbatches=100,
	threads=8, 
	sampling_mode="normal", 
	bern_flag=1, 
	filter_flag=1, 
	neg_ent=25,
	neg_rel=0
)

# Prepare file for meta
meta_rel_tot = 0
list_entity_meta = []
list_meta = []
batch_meta_size = 0

with open("./benchmarks/{}/meta/relation2id.txt".format(args.data), 'r') as f:
	meta_rel_tot = int(f.readline().strip())

if args.meta != 0.0:
	with open("./benchmarks/{}/meta/entity2id.txt".format(args.data), 'r') as f:
		f.readline()
		for line in f.readlines():
			triplet, _ = line.strip().split()
			h, t, r = triplet.strip().split('_')
			list_entity_meta.append([h, t, r])
	list_entity_meta = np.array(list_entity_meta).astype(int)

	with open("./benchmarks/{}/meta/train2id.txt".format(args.data), 'r') as f:
		f.readline()
		for line in f.readlines():
			h, t, r = line.strip().split()
			list_meta.append([h, t, r])
	list_meta = np.array(list_meta).astype(int)

	batch_meta_size = int(len(list_meta) / 100)

# Prepare file for aug
list_aug = []
batch_aug_size = 0

if args.aug != 0.0:
	with open("./benchmarks/{}/base/augment2id.txt".format(args.data), 'r') as f:
		f.readline()
		for line in f.readlines():
			h, t, r = line.strip().split()
			list_aug.append([h, t, r])
	list_aug = np.array(list_aug).astype(int)

	batch_aug_size = int(len(list_aug) / 100)

# define the model
BiVE_B = BiVE_B(
	ent_tot=train_dataloader.get_ent_tot(),
	rel_tot=train_dataloader.get_rel_tot(),
	meta_rel_tot=meta_rel_tot,
	dim=50,
	seed=args.serial
)

try:
	print("Loading checkpoint alpha_{}_regul_{}_meta_{}_aug_{}_epoch_{}-{}.ckpt".format(args.alpha, args.regul, args.meta, args.aug, args.epoch, args.serial))
	BiVE_B.load_checkpoint('./checkpoint/{}/BiVE_B/alpha_{}_regul_{}_meta_{}_aug_{}_epoch_{}-{}.ckpt'.format(args.data, args.alpha, args.regul, args.meta, args.aug, args.epoch, args.serial))
except:
	print("No checkpoint available!")
		
	# define the loss function
	model = NegativeSampling(
		model=BiVE_B, 
		loss=SoftplusLoss_submitted(),
		batch_size=train_dataloader.get_batch_size(), 
		regul_rate=args.regul,
		meta=(args.meta != 0.0),
		batch_meta_size=batch_meta_size,
		aug=(args.aug != 0.0),
		batch_aug_size=batch_aug_size
	)

	# train the model
	trainer = Trainer(
		model=model,
		data_loader=train_dataloader,
		train_times=args.epoch,
		alpha=args.alpha,
		use_gpu=True,
		opt_method="adagrad",
		list_entity_meta=list_entity_meta,
		list_meta=list_meta,
		batch_meta_size=batch_meta_size,
		weight_meta=args.meta,
		list_aug=list_aug,
		batch_aug_size=batch_aug_size,
		weight_aug=args.aug
	)
	
	trainer.run()
	os.makedirs('./checkpoint/{}/BiVE_B'.format(args.data), exist_ok=True)
	BiVE_B.save_checkpoint('./checkpoint/{}/BiVE_B/alpha_{}_regul_{}_meta_{}_aug_{}_epoch_{}-{}.ckpt'.format(args.data, args.alpha, args.regul, args.meta, args.aug, args.epoch, args.serial))

if args.lp:
	test_dataloader = TestDataLoader("./benchmarks/{}/base/".format(args.data), "link")
	tester = Tester(model=BiVE_B, data_loader=test_dataloader, use_gpu=True)
	mr, mrr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)

elif args.tp:
	test_dataloader = TestDataLoader("./benchmarks/{}/meta/".format(args.data), "link")
	tester = Tester(model=BiVE_B, data_loader=test_dataloader, use_gpu=True)
	mr, mrr, hit10, hit3, hit1 = tester.run_triplet_prediction(type_constrain=False, list_entity_meta=list_entity_meta)
	
elif args.clp:
	list_info = []
	with open("./benchmarks/{}/conditional/relation2id.txt".format(args.data), 'r') as f:
		f.readline()
		for line in f.readlines():
			relationship, _ = line.strip().split()
			h1, t1, r1, R, r2 = relationship.split('_')
			list_info.append((h1, t1, r1, R, r2))
	list_info = np.array(list_info).astype(int)

	test_dataloader = TestDataLoader("./benchmarks/{}/conditional/".format(args.data), "link")
	tester = Tester(model=BiVE_B, data_loader=test_dataloader, use_gpu=True)
	mr_tail, mrr_tail, hit10_tail, hit3_tail, hit1_tail = tester.run_conditional_link_prediction(type_constrain=False, list_info=list_info, weight_meta=args.meta)

	list_info = []
	with open("./benchmarks/{}/conditional_head/relation2id.txt".format(args.data), 'r') as f:
		f.readline()
		for line in f.readlines():
			relationship, _ = line.strip().split()
			r1, R, h2, t2, r2 = relationship.split('_')
			list_info.append((r1, R, h2, t2, r2))
	list_info = np.array(list_info).astype(int)

	test_dataloader = TestDataLoader("./benchmarks/{}/conditional_head/".format(args.data), "link")
	tester = Tester(model=BiVE_B, data_loader=test_dataloader, use_gpu=True)
	mr_head, mrr_head, hit10_head, hit3_head, hit1_head = tester.run_conditional_link_prediction_head(type_constrain=False, list_info=list_info, weight_meta=args.meta)
	mr, mrr, hit10, hit3, hit1 = (mr_tail + mr_head) / 2, (mrr_tail + mrr_head) / 2, (hit10_tail + hit10_head) / 2, (hit3_tail + hit3_head) / 2, (hit1_tail + hit1_head) / 2
	
if args.lp:
	os.makedirs("./result_test/{}/BiVE_B/LP".format(args.data), exist_ok=True)
	with open("./result_test/{}/BiVE_B/LP/alpha_{}_regul_{}_meta_{}_aug_{}_epoch_{}-{}.txt".format(args.data, args.alpha, args.regul, args.meta, args.aug, args.epoch, args.serial), 'w') as f:
		f.write("LP: {} {} {} {} {}\n".format(mr, mrr, hit10, hit3, hit1))

elif args.tp:
	os.makedirs("./result_test/{}/BiVE_B/TP".format(args.data), exist_ok=True)
	with open("./result_test/{}/BiVE_B/TP/alpha_{}_regul_{}_meta_{}_aug_{}_epoch_{}-{}.txt".format(args.data, args.alpha, args.regul, args.meta, args.aug, args.epoch, args.serial), 'w') as f:
		f.write("TP: {} {} {} {} {}\n".format(mr, mrr, hit10, hit3, hit1))

elif args.clp:
	os.makedirs("./result_test/{}/BiVE_B/CLP".format(args.data), exist_ok=True)
	with open("./result_test/{}/BiVE_B/CLP/alpha_{}_regul_{}_meta_{}_aug_{}_epoch_{}-{}.txt".format(args.data, args.alpha, args.regul, args.meta, args.aug, args.epoch, args.serial), 'w') as f:
		f.write("CLP: {} {} {} {} {}\n".format(mr, mrr, hit10, hit3, hit1))