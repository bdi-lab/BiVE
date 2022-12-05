import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState

class BiVE_B(Model):
	def __init__(self, ent_tot, rel_tot, meta_rel_tot, dim=50, seed=123):
		super(BiVE_B, self).__init__(ent_tot, rel_tot)

		self.meta_rel_tot = meta_rel_tot
		self.dim = dim
		self.seed = seed

		self.ent_s = nn.Embedding(self.ent_tot, self.dim)
		self.ent_x = nn.Embedding(self.ent_tot, self.dim)
		self.ent_y = nn.Embedding(self.ent_tot, self.dim)
		self.ent_z = nn.Embedding(self.ent_tot, self.dim)

		self.rel_p_s = nn.Embedding(self.rel_tot, self.dim)
		self.rel_p_x = nn.Embedding(self.rel_tot, self.dim)
		self.rel_p_y = nn.Embedding(self.rel_tot, self.dim)
		self.rel_p_z = nn.Embedding(self.rel_tot, self.dim)

		self.rel_s = nn.Embedding(self.rel_tot, self.dim)
		self.rel_x = nn.Embedding(self.rel_tot, self.dim)
		self.rel_y = nn.Embedding(self.rel_tot, self.dim)
		self.rel_z = nn.Embedding(self.rel_tot, self.dim)

		if self.meta_rel_tot > 0:
			self.meta_rel_p_s = nn.Embedding(self.meta_rel_tot, self.dim)
			self.meta_rel_p_x = nn.Embedding(self.meta_rel_tot, self.dim)
			self.meta_rel_p_y = nn.Embedding(self.meta_rel_tot, self.dim)
			self.meta_rel_p_z = nn.Embedding(self.meta_rel_tot, self.dim)

			self.meta_rel_s = nn.Embedding(self.meta_rel_tot, self.dim)
			self.meta_rel_x = nn.Embedding(self.meta_rel_tot, self.dim)
			self.meta_rel_y = nn.Embedding(self.meta_rel_tot, self.dim)
			self.meta_rel_z = nn.Embedding(self.meta_rel_tot, self.dim)

		self.W_s = nn.Linear(4 * self.dim, self.dim, bias=False)
		self.W_x = nn.Linear(4 * self.dim, self.dim, bias=False)
		self.W_y = nn.Linear(4 * self.dim, self.dim, bias=False)
		self.W_z = nn.Linear(4 * self.dim, self.dim, bias=False)

		self.init_weights()

	def init_weights(self):
		s, x, y, z = self.quaternion_init(self.ent_tot, self.dim)
		s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
		self.ent_s.weight.data = s.type_as(self.ent_s.weight.data)
		self.ent_x.weight.data = x.type_as(self.ent_x.weight.data)
		self.ent_y.weight.data = y.type_as(self.ent_y.weight.data)
		self.ent_z.weight.data = z.type_as(self.ent_z.weight.data)

		s, x, y, z = self.quaternion_init(self.rel_tot, self.dim)
		s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
		self.rel_p_s.weight.data = s.type_as(self.rel_p_s.weight.data)
		self.rel_p_x.weight.data = x.type_as(self.rel_p_x.weight.data)
		self.rel_p_y.weight.data = y.type_as(self.rel_p_y.weight.data)
		self.rel_p_z.weight.data = z.type_as(self.rel_p_z.weight.data)

		s, x, y, z = self.quaternion_init(self.rel_tot, self.dim)
		s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
		self.rel_s.weight.data = s.type_as(self.rel_s.weight.data)
		self.rel_x.weight.data = x.type_as(self.rel_x.weight.data)
		self.rel_y.weight.data = y.type_as(self.rel_y.weight.data)
		self.rel_z.weight.data = z.type_as(self.rel_z.weight.data)

		if self.meta_rel_tot > 0:
			s, x, y, z = self.quaternion_init(self.meta_rel_tot, self.dim)
			s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
			self.meta_rel_p_s.weight.data = s.type_as(self.meta_rel_p_s.weight.data)
			self.meta_rel_p_x.weight.data = x.type_as(self.meta_rel_p_x.weight.data)
			self.meta_rel_p_y.weight.data = y.type_as(self.meta_rel_p_y.weight.data)
			self.meta_rel_p_z.weight.data = z.type_as(self.meta_rel_p_z.weight.data)

			s, x, y, z = self.quaternion_init(self.meta_rel_tot, self.dim)
			s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
			self.meta_rel_s.weight.data = s.type_as(self.meta_rel_s.weight.data)
			self.meta_rel_x.weight.data = x.type_as(self.meta_rel_x.weight.data)
			self.meta_rel_y.weight.data = y.type_as(self.meta_rel_y.weight.data)
			self.meta_rel_z.weight.data = z.type_as(self.meta_rel_z.weight.data)

		nn.init.xavier_uniform_(self.W_s.weight.data)
		nn.init.xavier_uniform_(self.W_x.weight.data)
		nn.init.xavier_uniform_(self.W_y.weight.data)
		nn.init.xavier_uniform_(self.W_z.weight.data)

	def quaternion_init(self, in_features, out_features, criterion='he'):
		fan_in = in_features
		fan_out = out_features

		if criterion == 'glorot':
			s = 1. / np.sqrt(2 * (fan_in + fan_out))
		elif criterion == 'he':
			s = 1. / np.sqrt(2 * fan_in)
		else:
			raise ValueError('Invalid criterion: ', criterion)
		rng = RandomState(self.seed)

		# Generating randoms and purely imaginary quaternions :
		kernel_shape = (in_features, out_features)

		number_of_weights = np.prod(kernel_shape)
		v_i = np.random.uniform(0.0, 1.0, number_of_weights)
		v_j = np.random.uniform(0.0, 1.0, number_of_weights)
		v_k = np.random.uniform(0.0, 1.0, number_of_weights)

		# Purely imaginary quaternions unitary
		for i in range(0, number_of_weights):
			norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
			v_i[i] /= norm
			v_j[i] /= norm
			v_k[i] /= norm
		v_i = v_i.reshape(kernel_shape)
		v_j = v_j.reshape(kernel_shape)
		v_k = v_k.reshape(kernel_shape)

		modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
		phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

		weight_r = modulus * np.cos(phase)
		weight_i = modulus * v_i * np.sin(phase)
		weight_j = modulus * v_j * np.sin(phase)
		weight_k = modulus * v_k * np.sin(phase)

		return (weight_r, weight_i, weight_j, weight_k)

	def _calc(self, s_h, x_h, y_h, z_h, s_t, x_t, y_t, z_t, s_r, x_r, y_r, z_r):
		denominator_r = torch.sqrt(s_r ** 2 + x_r ** 2 + y_r ** 2 + z_r ** 2)
		s_r = s_r / denominator_r
		x_r = x_r / denominator_r
		y_r = y_r / denominator_r
		z_r = z_r / denominator_r

		A = s_h * s_r - x_h * x_r - y_h * y_r - z_h * z_r
		B = s_h * x_r + s_r * x_h + y_h * z_r - y_r * z_h
		C = s_h * y_r + s_r * y_h + z_h * x_r - z_r * x_h
		D = s_h * z_r + s_r * z_h + x_h * y_r - x_r * y_h

		score_r = (A * s_t + B * x_t + C * y_t + D * z_t)
		return -torch.sum(score_r, -1)

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		s_h = self.ent_s(batch_h)
		x_h = self.ent_x(batch_h)
		y_h = self.ent_y(batch_h)
		z_h = self.ent_z(batch_h)

		s_t = self.ent_s(batch_t)
		x_t = self.ent_x(batch_t)
		y_t = self.ent_y(batch_t)
		z_t = self.ent_z(batch_t)

		s_r_p = self.rel_p_s(batch_r)
		x_r_p = self.rel_p_x(batch_r)
		y_r_p = self.rel_p_y(batch_r)
		z_r_p = self.rel_p_z(batch_r)

		s_r = self.rel_s(batch_r)
		x_r = self.rel_x(batch_r)
		y_r = self.rel_y(batch_r)
		z_r = self.rel_z(batch_r)

		score = self._calc(s_h + s_r_p, x_h + x_r_p, y_h + y_r_p, z_h + z_r_p, s_t, x_t, y_t, z_t, s_r, x_r, y_r, z_r)
		return score

	def forward_meta(self, data):
		batch_h1 = data['batch_h1']
		batch_t1 = data['batch_t1']
		batch_r1 = data['batch_r1']
		batch_R = data['batch_R']
		batch_h2 = data['batch_h2']
		batch_t2 = data['batch_t2']
		batch_r2 = data['batch_r2']

		s_h1 = self.ent_s(batch_h1)
		x_h1 = self.ent_x(batch_h1)
		y_h1 = self.ent_y(batch_h1)
		z_h1 = self.ent_z(batch_h1)

		s_t1 = self.ent_s(batch_t1)
		x_t1 = self.ent_x(batch_t1)
		y_t1 = self.ent_y(batch_t1)
		z_t1 = self.ent_z(batch_t1)

		s_r1_p = self.rel_p_s(batch_r1)
		x_r1_p = self.rel_p_x(batch_r1)
		y_r1_p = self.rel_p_y(batch_r1)
		z_r1_p = self.rel_p_z(batch_r1)

		s_r1 = self.rel_s(batch_r1)
		x_r1 = self.rel_x(batch_r1)
		y_r1 = self.rel_y(batch_r1)
		z_r1 = self.rel_z(batch_r1)

		s_h2 = self.ent_s(batch_h2)
		x_h2 = self.ent_x(batch_h2)
		y_h2 = self.ent_y(batch_h2)
		z_h2 = self.ent_z(batch_h2)

		s_t2 = self.ent_s(batch_t2)
		x_t2 = self.ent_x(batch_t2)
		y_t2 = self.ent_y(batch_t2)
		z_t2 = self.ent_z(batch_t2)

		s_r2_p = self.rel_p_s(batch_r2)
		x_r2_p = self.rel_p_x(batch_r2)
		y_r2_p = self.rel_p_y(batch_r2)
		z_r2_p = self.rel_p_z(batch_r2)

		s_r2 = self.rel_s(batch_r2)
		x_r2 = self.rel_x(batch_r2)
		y_r2 = self.rel_y(batch_r2)
		z_r2 = self.rel_z(batch_r2)

		s_R_p = self.meta_rel_p_s(batch_R)
		x_R_p = self.meta_rel_p_x(batch_R)
		y_R_p = self.meta_rel_p_y(batch_R)
		z_R_p = self.meta_rel_p_z(batch_R)

		s_R = self.meta_rel_s(batch_R)
		x_R = self.meta_rel_x(batch_R)
		y_R = self.meta_rel_y(batch_R)
		z_R = self.meta_rel_z(batch_R)

		s_H = self.W_s(torch.cat((s_h1, s_r1, s_r1_p, s_t1), 1))
		x_H = self.W_x(torch.cat((x_h1, x_r1, x_r1_p, x_t1), 1))
		y_H = self.W_y(torch.cat((y_h1, y_r1, y_r1_p, y_t1), 1))
		z_H = self.W_z(torch.cat((z_h1, z_r1, z_r1_p, z_t1), 1))

		s_T = self.W_s(torch.cat((s_h2, s_r2, s_r2_p, s_t2), 1))
		x_T = self.W_x(torch.cat((x_h2, x_r2, x_r2_p, x_t2), 1))
		y_T = self.W_y(torch.cat((y_h2, y_r2, y_r2_p, y_t2), 1))
		z_T = self.W_z(torch.cat((z_h2, z_r2, z_r2_p, z_t2), 1))

		score_meta = self._calc(s_H + s_R_p, x_H + x_R_p, y_H + y_R_p, z_H + z_R_p, s_T, x_T, y_T, z_T, s_R, x_R, y_R, z_R)
		return score_meta

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		s_h = self.ent_s(batch_h)
		x_h = self.ent_x(batch_h)
		y_h = self.ent_y(batch_h)
		z_h = self.ent_z(batch_h)

		s_t = self.ent_s(batch_t)
		x_t = self.ent_x(batch_t)
		y_t = self.ent_y(batch_t)
		z_t = self.ent_z(batch_t)

		s_r_p = self.rel_p_s(batch_r)
		x_r_p = self.rel_p_x(batch_r)
		y_r_p = self.rel_p_y(batch_r)
		z_r_p = self.rel_p_z(batch_r)

		s_r = self.rel_s(batch_r)
		x_r = self.rel_x(batch_r)
		y_r = self.rel_y(batch_r)
		z_r = self.rel_z(batch_r)
		
		regul = (torch.mean(torch.abs(s_h) ** 2)
				+ torch.mean(torch.abs(x_h) ** 2)
				+ torch.mean(torch.abs(y_h) ** 2)
				+ torch.mean(torch.abs(z_h) ** 2)
				+ torch.mean(torch.abs(s_t) ** 2)
				+ torch.mean(torch.abs(x_t) ** 2)
				+ torch.mean(torch.abs(y_t) ** 2)
				+ torch.mean(torch.abs(z_t) ** 2))

		regul2 = (torch.mean(torch.abs(s_r_p) ** 2)
				+ torch.mean(torch.abs(x_r_p) ** 2)
				+ torch.mean(torch.abs(y_r_p) ** 2)
				+ torch.mean(torch.abs(z_r_p) ** 2)
				+ torch.mean(torch.abs(s_r) ** 2)
				+ torch.mean(torch.abs(x_r) ** 2)
				+ torch.mean(torch.abs(y_r) ** 2)
				+ torch.mean(torch.abs(z_r) ** 2))
		return regul + regul2

	def regularization_meta(self, data):
		batch_h1 = data['batch_h1']
		batch_t1 = data['batch_t1']
		batch_r1 = data['batch_r1']
		batch_R = data['batch_R']
		batch_h2 = data['batch_h2']
		batch_t2 = data['batch_t2']
		batch_r2 = data['batch_r2']

		s_h1 = self.ent_s(batch_h1)
		x_h1 = self.ent_x(batch_h1)
		y_h1 = self.ent_y(batch_h1)
		z_h1 = self.ent_z(batch_h1)

		s_t1 = self.ent_s(batch_t1)
		x_t1 = self.ent_x(batch_t1)
		y_t1 = self.ent_y(batch_t1)
		z_t1 = self.ent_z(batch_t1)

		s_r1_p = self.rel_p_s(batch_r1)
		x_r1_p = self.rel_p_x(batch_r1)
		y_r1_p = self.rel_p_y(batch_r1)
		z_r1_p = self.rel_p_z(batch_r1)

		s_r1 = self.rel_s(batch_r1)
		x_r1 = self.rel_x(batch_r1)
		y_r1 = self.rel_y(batch_r1)
		z_r1 = self.rel_z(batch_r1)

		s_h2 = self.ent_s(batch_h2)
		x_h2 = self.ent_x(batch_h2)
		y_h2 = self.ent_y(batch_h2)
		z_h2 = self.ent_z(batch_h2)

		s_t2 = self.ent_s(batch_t2)
		x_t2 = self.ent_x(batch_t2)
		y_t2 = self.ent_y(batch_t2)
		z_t2 = self.ent_z(batch_t2)

		s_r2_p = self.rel_p_s(batch_r2)
		x_r2_p = self.rel_p_x(batch_r2)
		y_r2_p = self.rel_p_y(batch_r2)
		z_r2_p = self.rel_p_z(batch_r2)

		s_r2 = self.rel_s(batch_r2)
		x_r2 = self.rel_x(batch_r2)
		y_r2 = self.rel_y(batch_r2)
		z_r2 = self.rel_z(batch_r2)

		s_R_p = self.meta_rel_p_s(batch_R)
		x_R_p = self.meta_rel_p_x(batch_R)
		y_R_p = self.meta_rel_p_y(batch_R)
		z_R_p = self.meta_rel_p_z(batch_R)

		s_R = self.meta_rel_s(batch_R)
		x_R = self.meta_rel_x(batch_R)
		y_R = self.meta_rel_y(batch_R)
		z_R = self.meta_rel_z(batch_R)

		s_H = self.W_s(torch.cat((s_h1, s_r1, s_r1_p, s_t1), 1))
		x_H = self.W_x(torch.cat((x_h1, x_r1, x_r1_p, x_t1), 1))
		y_H = self.W_y(torch.cat((y_h1, y_r1, y_r1_p, y_t1), 1))
		z_H = self.W_z(torch.cat((z_h1, z_r1, z_r1_p, z_t1), 1))

		s_T = self.W_s(torch.cat((s_h2, s_r2, s_r2_p, s_t2), 1))
		x_T = self.W_x(torch.cat((x_h2, x_r2, x_r2_p, x_t2), 1))
		y_T = self.W_y(torch.cat((y_h2, y_r2, y_r2_p, y_t2), 1))
		z_T = self.W_z(torch.cat((z_h2, z_r2, z_r2_p, z_t2), 1))
		
		regul = (torch.mean(torch.abs(s_H) ** 2)
				+ torch.mean(torch.abs(x_H) ** 2)
				+ torch.mean(torch.abs(y_H) ** 2)
				+ torch.mean(torch.abs(z_H) ** 2)
				+ torch.mean(torch.abs(s_T) ** 2)
				+ torch.mean(torch.abs(x_T) ** 2)
				+ torch.mean(torch.abs(y_T) ** 2)
				+ torch.mean(torch.abs(z_T) ** 2))

		regul2 = (torch.mean(torch.abs(s_R_p) ** 2)
				+ torch.mean(torch.abs(x_R_p) ** 2)
				+ torch.mean(torch.abs(y_R_p) ** 2)
				+ torch.mean(torch.abs(z_R_p) ** 2)
				+ torch.mean(torch.abs(s_R) ** 2)
				+ torch.mean(torch.abs(x_R) ** 2)
				+ torch.mean(torch.abs(y_R) ** 2)
				+ torch.mean(torch.abs(z_R) ** 2))
		return regul + regul2

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()

	def predict_meta(self, data):
		score = -self.forward_meta(data)
		return score.cpu().data.numpy()