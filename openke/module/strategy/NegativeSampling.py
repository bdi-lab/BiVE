from .Strategy import Strategy

class NegativeSampling(Strategy):

	def __init__(self, model=None, loss=None, batch_size=256, regul_rate=0.0, l3_regul_rate=0.0,
				 meta=False, batch_meta_size=256, aug=False, batch_aug_size=256):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate
		self.meta = meta
		self.batch_meta_size = batch_meta_size
		self.aug = aug
		self.batch_aug_size = batch_aug_size

	def _get_positive_score(self, score, batch_type='base'):
		if batch_type == 'base':
			positive_score = score[:self.batch_size]
			positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		elif batch_type == 'meta':
			positive_score = score[:self.batch_meta_size]
			positive_score = positive_score.view(-1, self.batch_meta_size).permute(1, 0)
		elif batch_type == 'aug':
			positive_score = score[:self.batch_aug_size]
			positive_score = positive_score.view(-1, self.batch_aug_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score, batch_type='base'):
		if batch_type == 'base':
			negative_score = score[self.batch_size:]
			negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		elif batch_type == 'meta':
			negative_score = score[self.batch_meta_size:]
			negative_score = negative_score.view(-1, self.batch_meta_size).permute(1, 0)
		elif batch_type == 'aug':
			negative_score = score[self.batch_aug_size:]
			negative_score = negative_score.view(-1, self.batch_aug_size).permute(1, 0)
		return negative_score

	def forward(self, data, batch_type='base'):
		if batch_type == 'base' or batch_type == 'aug':
			score = self.model.forward(data)
		elif batch_type == 'meta':
			score = self.model.forward_meta(data)
		p_score = self._get_positive_score(score, batch_type)
		n_score = self._get_negative_score(score, batch_type)
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			if batch_type == 'base' or batch_type == 'aug':
				loss_res += self.regul_rate * self.model.regularization(data)
			elif batch_type == 'meta':
				loss_res += self.regul_rate * self.model.regularization_meta(data)
		if self.l3_regul_rate != 0:
			if batch_type == 'base' or batch_type == 'aug':
				loss_res += self.l3_regul_rate * self.model.l3_regularization()
			elif batch_type == 'meta':
				loss_res += self.l3_regul_rate * self.model.l3_regularization_meta()
		return loss_res