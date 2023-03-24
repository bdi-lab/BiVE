# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm

class Trainer(object):

	def __init__(self, 
				 model=None,
				 data_loader=None,
				 train_times=500,
				 alpha=0.5,
				 use_gpu=True,
				 opt_method="sgd",
				 save_steps=None,
				 checkpoint_dir=None,
				 list_entity_meta=None,
				 list_meta=None,
				 batch_meta_size=256,
				 weight_meta=0.0,
				 list_aug=None,
				 batch_aug_size=256,
				 weight_aug=0.0):

		self.work_threads = 8
		self.train_times = train_times

		self.opt_method = opt_method
		self.optimizer = None
		self.lr_decay = 0
		self.weight_decay = 0
		self.alpha = alpha

		self.model = model
		self.data_loader = data_loader
		self.use_gpu = use_gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir

		self.list_entity_meta = list_entity_meta
		self.list_meta = list_meta
		self.batch_meta_size = batch_meta_size
		self.weight_meta = weight_meta

		self.list_aug = list_aug
		self.batch_aug_size = batch_aug_size
		self.weight_aug = weight_aug

	def train_one_step(self, data):
		self.optimizer.zero_grad()
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode']
		}, 'base')

		if self.weight_meta != 0.0:
			loss_meta = self.model({
				'batch_h1': self.to_var(data['batch_h1'], self.use_gpu),
				'batch_t1': self.to_var(data['batch_t1'], self.use_gpu),
				'batch_r1': self.to_var(data['batch_r1'], self.use_gpu),
				'batch_R': self.to_var(data['batch_R'], self.use_gpu),
				'batch_h2': self.to_var(data['batch_h2'], self.use_gpu),
				'batch_t2': self.to_var(data['batch_t2'], self.use_gpu),
				'batch_r2': self.to_var(data['batch_r2'], self.use_gpu),
				'batch_Y': self.to_var(data['batch_Y'], self.use_gpu),
				'mode': data['mode'],
			}, 'meta')
			loss += self.weight_meta * self.batch_meta_size / self.data_loader.get_batch_size() * loss_meta

		if self.weight_aug != 0.0:
			loss_aug = self.model({
				'batch_h': self.to_var(data['batch_h_aug'], self.use_gpu),
				'batch_t': self.to_var(data['batch_t_aug'], self.use_gpu),
				'batch_r': self.to_var(data['batch_r_aug'], self.use_gpu),
				'batch_y': self.to_var(data['batch_y_aug'], self.use_gpu),
				'mode': data['mode'],
			}, 'aug')
			loss += self.weight_aug * self.batch_aug_size / self.data_loader.get_batch_size() * loss_aug
		loss.backward()
		self.optimizer.step()		 
		return loss.item()

	def run(self):
		if self.use_gpu:
			self.model.cuda()

		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")
		
		training_range = tqdm(range(self.train_times))
		for epoch in training_range:
			res = 0.0
			for data in self.data_loader:
				if self.weight_meta != 0.0:
					data = self.sampling_meta(data)
				if self.weight_aug != 0.0:
					data = self.sampling_aug(data)
				loss = self.train_one_step(data)
				res += loss
			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			
			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

	def sampling_meta(self, data):
		batch_meta = np.random.choice(len(self.list_meta), self.batch_meta_size)
		batch_meta = self.list_meta[batch_meta].copy()
		batch_meta = np.tile(batch_meta, (26, 1))

		corrupt = np.random.choice(2, 25 * self.batch_meta_size)
		corrupt_target = np.random.choice(len(self.list_entity_meta), 25 * self.batch_meta_size)
		batch_meta[np.where(corrupt == 0)[0] + self.batch_meta_size, 0] = corrupt_target[np.where(corrupt == 0)]
		batch_meta[np.where(corrupt == 1)[0] + self.batch_meta_size, 1] = corrupt_target[np.where(corrupt == 1)]

		batch_y = np.zeros(self.batch_meta_size * 26)
		batch_y[self.batch_meta_size:] = -1
		data['batch_h1'], data['batch_t1'], data['batch_r1'] = self.list_entity_meta[batch_meta.transpose()[0]].transpose()
		data['batch_h2'], data['batch_t2'], data['batch_r2'] = self.list_entity_meta[batch_meta.transpose()[1]].transpose()
		data['batch_R'] = batch_meta.transpose()[2]
		data['batch_Y'] = batch_y

		return data

	def sampling_aug(self, data):
		batch_aug = np.random.choice(len(self.list_aug), self.batch_aug_size)
		batch_aug = self.list_aug[batch_aug].copy()
		batch_aug = np.tile(batch_aug, (26, 1))

		corrupt = np.random.choice(2, 25 * self.batch_aug_size)
		corrupt_target = np.random.choice(self.model.model.ent_tot, 25 * self.batch_aug_size)
		batch_aug[np.where(corrupt == 0)[0] + self.batch_aug_size, 0] = corrupt_target[np.where(corrupt == 0)]
		batch_aug[np.where(corrupt == 1)[0] + self.batch_aug_size, 1] = corrupt_target[np.where(corrupt == 1)]

		batch_y = np.zeros(self.batch_aug_size * 26)
		batch_y[self.batch_aug_size:] = -1
		data['batch_h_aug'], data['batch_t_aug'], data['batch_r_aug'] = batch_aug.transpose()
		data['batch_y_aug'] = batch_y

		return data

	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir