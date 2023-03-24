import random
from tqdm import tqdm
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument('conf', type=float)
parser.add_argument('--count', type=int, default=0)
parser.add_argument('--num_iter', type=int, default=50000000)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

### Set random seed ###
import torch
import numpy as np
import random

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.use_deterministic_algorithms(True)

### Function: remove_duplicate
def remove_duplicate(target):
	return list(dict.fromkeys(target))

### Load base triplets ###
train = []
with open("./benchmarks/{}/base/train2id.txt".format(args.data), 'r') as f:
	f.readline()
	for line in f.readlines():
		h, t, r = line.strip().split()
		train.append((h, r, t))
remove_duplicate(train)

### Load higher-order triplets ###
idx2triplet = dict()
with open("./benchmarks/{}/meta/entity2id.txt".format(args.data), 'r') as f:
	f.readline()
	for line in f.readlines():
		triplet, idx = line.strip().split()
		h, t, r = triplet.split('_')
		idx2triplet[idx] = (h, r, t)

train_ho = []
with open("./benchmarks/{}/meta/train2id.txt".format(args.data), 'r') as f:
	f.readline()
	for line in f.readlines():
		H, T, R = line.strip().split()
		train_ho.append((idx2triplet[H], R, idx2triplet[T]))
remove_duplicate(train_ho)

### Sanity check ###
print("Number of base triplet:", len(train))
print("Number of higher-order triplet:", len(train_ho))

### Function: inv
def inv(r):
	return r + '_inv'

### Merge triplets into base form ###
### triplet = [(h, r, t, visited)]
triplet = []
for h, r, t in train:
	triplet.append((h, r, t, (t, )))
	triplet.append((t, inv(r), h, (h, )))

for (h1, r1, t1), R, (h2, r2, t2) in train_ho:
	triplet.append((h1, '/'.join([r1, R, inv(r2)]), h2, (t1, h2, t2)))
	triplet.append((h1, '/'.join([r1, R, r2]), t2, (t1, h2, t2)))
	triplet.append((t1, '/'.join([inv(r1), R, inv(r2)]), h2, (h1, h2, t2)))
	triplet.append((t1, '/'.join([inv(r1), R, r2]), t2, (h1, h2, t2)))
	triplet.append((h2, '/'.join([inv(r1), inv(R), r2]), h1, (h1, t1, t2)))
	triplet.append((h2, '/'.join([r1, inv(R), r2]), t1, (h1, t1, t2)))
	triplet.append((t2, '/'.join([inv(r1), inv(R), inv(r2)]), h1, (h1, t1, h2)))
	triplet.append((t2, '/'.join([r1, inv(R), inv(r2)]), t1, (h1, t1, h2)))
remove_duplicate(triplet)

### Sanity check ###
assert len(triplet) == 2 * len(train) + 8 * len(train_ho)
print("Number of triplet:", len(triplet))

### Generate candidates for random walk ###
candidate = dict()
for h, r, t, visited in triplet:
	if h not in candidate:
		candidate[h] = []
	candidate[h].append((r, t, visited))
entity = list(candidate.keys())

### Function: Random walk ###
def random_walk(length):
	### Starting entity
	cur = random.choice(entity)
	visited = {cur}
	relation_path = []
	path_instance = [cur]
	for i in range(length):
		r, t, new_visit = random.choice(candidate[cur])
		
		### Reached to already visited entity ###
		if t in visited:
			return False

		### Update states ###
		relation_path += [r]
		path_instance += [r, t]
		visited = visited.union(new_visit)
		cur = t
	return tuple(relation_path), tuple(path_instance)

### Collect random walks ###
instances = dict()

start = time.time()
print("Collecting random walks with length 2")
cnt = 0
while cnt < args.num_iter:
	res = random_walk(2)
	if res == False:
		continue
	relation_path, path_instance = res

	if relation_path not in instances:
		instances[relation_path] = []
	instances[relation_path].append(path_instance)
	cnt += 1
	if cnt % 1000000 == 0:
		print("Collecting {} walks ({:.6f} s)".format(cnt, time.time() - start))
print()
start = time.time()
print("Collecting random walks with length 3")
cnt = 0
while cnt < args.num_iter:
	res = random_walk(3)
	if res == False:
		continue
	relation_path, path_instance = res

	if relation_path not in instances:
		instances[relation_path] = []
	instances[relation_path].append(path_instance)
	cnt += 1
	if cnt % 1000000 == 0:
		print("Collecting {} walks ({:.6f} s)".format(cnt, time.time() - start))

### Remove duplicate instances ###
for relation_path in instances:
	instances[relation_path] = remove_duplicate(instances[relation_path])

### Sanity check ###
cnt_path = 0
cnt_instance = 0
for relation_path in tqdm(instances):
	cnt_path += 1
	cnt_instance += len(instances[relation_path])
print("Number of relation path:", cnt_path)
print("Number of instances:", cnt_instance)

### Create head-tail dictionary ###
relation = dict()
for h, r, t in train:
	if (h, t) not in relation:
		relation[(h, t)] = []
	relation[(h, t)].append(r)
	if (t, h) not in relation:
		relation[(t, h)] = []
	relation[(t, h)].append(inv(r))

### Remove relation paths with < 100 instances (for efficiency) ###
instances = dict(filter(lambda item: len(item[1]) >= args.count, instances.items()))
print("Removed relation paths with instances < {}".format(args.count))

### Count number of instances of relation paths ###
count = dict()
for relation_path in instances:
	count[relation_path] = len(instances[relation_path])

### Count number of patterns ###
count_pattern = dict()
for relation_path in tqdm(instances):
	count_pattern[relation_path] = dict()
	for path_instance in instances[relation_path]:
		h, t = path_instance[0], path_instance[-1]
		if (h, t) not in relation:
			continue
		for r in relation[(h, t)]:
			if r not in count_pattern[relation_path]:
				count_pattern[relation_path][r] = 0
			count_pattern[relation_path][r] += 1

cnt = 0
for relation_path in count_pattern:
	for r in count_pattern[relation_path]:
		cnt += 1
print("Number of (p, r):", cnt)

### Calculate frequency ###
frequency = dict()
for relation_path in tqdm(count):
	for r in count_pattern[relation_path]:
		if count_pattern[relation_path][r] < args.count:
			continue
		freq = count_pattern[relation_path][r] / count[relation_path]
		if freq < args.conf:
			continue
		frequency[(relation_path, r)] = (freq, count_pattern[relation_path][r], count[relation_path])

### Sort results by frequency ###
sort_frequency = sorted(frequency.items(), key=lambda item: item[1][0], reverse=True)

set_train = set(train)
list_aug = []
for (relation_path, relation), count in tqdm(sort_frequency):
	tmp_supp = []
	tmp_aug = []
	for instance in instances[relation_path]:
		h, t = instance[0], instance[-1]
		if relation.endswith('_inv'):
			if (t, relation[:-4], h) not in set_train:
				tmp_aug.append((t, relation[:-4], h))
			else:
				tmp_supp.append((t, relation[:-4], h))
		else:
			if (h, relation, t) not in set_train:
				tmp_aug.append((h, relation, t))
			else:
				tmp_supp.append((h, relation, t))
	tmp_supp = remove_duplicate(tmp_supp)
	tmp_aug = remove_duplicate(tmp_aug)
	list_aug += tmp_aug

list_aug = remove_duplicate(list_aug)
with open("./benchmarks/{}/base/augment2id.txt".format(args.data), 'w') as f:
	f.write("{}\n".format(len(list_aug)))
	for h, r, t in list_aug:
		f.write("{} {} {}\n".format(h, t, r))
print("Done writing augment2id.txt")