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
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm

class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })

    def run_link_prediction(self, type_constrain=False, save_dir=None, write_dir=None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        if save_dir != None:
            file = open(save_dir, 'w')
            cache = set()

        if write_dir != None:
            f = open(write_dir, 'w')

        for index, [data_head, data_tail] in enumerate(training_range):
            h, r, t = data_tail["batch_h"][0], data_head["batch_r"][0], data_head["batch_t"][0]
            score = self.test_one_step(data_head)
            if write_dir != None:
                tmp = score.argsort()
                rank = np.where(tmp == h)[0][0]
                if rank < 3:
                    f.write("_ {} {} | {}\n".format(r, t, h))
                    f.write("{}\n".format(' '.join(map(str, list(tmp[:rank + 1])))))
            if save_dir != None and ('_', r, t) not in cache:
                score_write = []
                for item in score:
                    score_write.append(str(item))
                file.write("_,{},{},{}\n".format(r, t, ','.join(score_write)))
                cache.add(('_', r, t))
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step(data_tail)
            if write_dir != None:
                tmp = score.argsort()
                rank = np.where(tmp == t)[0][0]
                if rank < 3:
                    f.write("{} {} _ | {}\n".format(h, r, t))
                    f.write("{}\n".format(' '.join(map(str, list(tmp[:rank + 1])))))
            if save_dir != None and (h, r, '_') not in cache:
                score_write = []
                for item in score:
                    score_write.append(str(item))
                file.write("{},{},_,{}\n".format(h, r, ','.join(score_write)))
                cache.add((h, r, '_'))
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain)

        if save_dir != None:
            file.close()

        if write_dir != None:
            f.close()

        mr = self.lib.getTestLinkMR(type_constrain)
        mrr = self.lib.getTestLinkMRR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print (hit10)
        return mr, mrr, hit10, hit3, hit1

    def test_one_step_meta(self, data, list_entity_meta):
        data['batch_h1'], data['batch_t1'], data['batch_r1'] = list_entity_meta[data['batch_h']].transpose()
        data['batch_h2'], data['batch_t2'], data['batch_r2'] = list_entity_meta[data['batch_t']].transpose()
        data['batch_R'] = data['batch_r']
        return self.model.predict_meta({
            'batch_h1': self.to_var(data['batch_h1'], self.use_gpu),
            'batch_t1': self.to_var(data['batch_t1'], self.use_gpu),
            'batch_r1': self.to_var(data['batch_r1'], self.use_gpu),
            'batch_R': self.to_var(data['batch_R'], self.use_gpu),
            'batch_h2': self.to_var(data['batch_h2'], self.use_gpu),
            'batch_t2': self.to_var(data['batch_t2'], self.use_gpu),
            'batch_r2': self.to_var(data['batch_r2'], self.use_gpu),
            'mode': data['mode'],
        })

    def run_triplet_prediction(self, type_constrain=False, list_entity_meta=None, save_dir=None, write_dir=None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        ranks = []
        if save_dir != None:
            f = open(save_dir, 'w')
        for index, [data_head, data_tail] in enumerate(training_range):
            h, r, t = data_tail["batch_h"][0], data_head["batch_r"][0], data_head["batch_t"][0]
            score = self.test_one_step_meta(data_head, list_entity_meta)
            if save_dir != None:
                tmp = score.argsort()
                rank = np.where(tmp == h)[0][0]
                if rank < 3:
                    f.write("_ {} {} | {}\n".format(r, t, h))
                    f.write("{}\n".format(' '.join(map(str, list(tmp[:rank + 1])))))
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            ranks.append(np.sum(score <= score[h]))
            score = self.test_one_step_meta(data_tail, list_entity_meta)
            if save_dir != None:
                tmp = score.argsort()
                rank = np.where(tmp == t)[0][0]
                if rank < 3:
                    f.write("{} {} _ | {}\n".format(h, r, t))
                    f.write("{}\n".format(' '.join(map(str, list(tmp[:rank + 1])))))
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
            ranks.append(np.sum(score <= score[t]))
        self.lib.test_link_prediction(type_constrain)
        print(np.mean(ranks))
        mr = self.lib.getTestLinkMR(type_constrain)
        mrr = self.lib.getTestLinkMRR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print (hit10)
        return mr, mrr, hit10, hit3, hit1

    def test_one_step_conditional_head(self, data, list_info, weight_meta):
        data['batch_h1'], data['batch_t1'], data['batch_r1'], data['batch_R'], data['batch_r2'] = list_info[data['batch_r']].transpose()
        data['batch_h2'] = data['batch_h']
        data['batch_t2'] = data['batch_t']
        data['batch_r'] = data['batch_r2']
        data['batch_t2'] = np.tile(data['batch_t2'], len(data['batch_h2']))
        data['batch_r2'] = np.tile(data['batch_r2'], len(data['batch_h2']))
        
        score = self.model.predict({
                'batch_h': self.to_var(data['batch_h'], self.use_gpu),
                'batch_t': self.to_var(data['batch_t'], self.use_gpu),
                'batch_r': self.to_var(data['batch_r'], self.use_gpu),
                'mode': data['mode'],
        })

        score_meta = self.model.predict_meta({
                'batch_h1': self.to_var(data['batch_h1'], self.use_gpu),
                'batch_t1': self.to_var(data['batch_t1'], self.use_gpu),
                'batch_r1': self.to_var(data['batch_r1'], self.use_gpu),
                'batch_R': self.to_var(data['batch_R'], self.use_gpu),
                'batch_h2': self.to_var(data['batch_h2'], self.use_gpu),
                'batch_t2': self.to_var(data['batch_t2'], self.use_gpu),
                'batch_r2': self.to_var(data['batch_r2'], self.use_gpu),
                'mode': data['mode'],
        })

        return score + weight_meta * score_meta

    def test_one_step_conditional_tail(self, data, list_info, weight_meta):
        data['batch_h1'], data['batch_t1'], data['batch_r1'], data['batch_R'], data['batch_r2'] = list_info[data['batch_r']].transpose()
        data['batch_h2'] = data['batch_h']
        data['batch_t2'] = data['batch_t']
        data['batch_r'] = data['batch_r2']
        data['batch_h2'] = np.tile(data['batch_h2'], len(data['batch_t2']))
        data['batch_r2'] = np.tile(data['batch_r2'], len(data['batch_t2']))
        
        score = self.model.predict({
                'batch_h': self.to_var(data['batch_h'], self.use_gpu),
                'batch_t': self.to_var(data['batch_t'], self.use_gpu),
                'batch_r': self.to_var(data['batch_r'], self.use_gpu),
                'mode': data['mode']
        })

        score_meta = self.model.predict_meta({
                'batch_h1': self.to_var(data['batch_h1'], self.use_gpu),
                'batch_t1': self.to_var(data['batch_t1'], self.use_gpu),
                'batch_r1': self.to_var(data['batch_r1'], self.use_gpu),
                'batch_R': self.to_var(data['batch_R'], self.use_gpu),
                'batch_h2': self.to_var(data['batch_h2'], self.use_gpu),
                'batch_t2': self.to_var(data['batch_t2'], self.use_gpu),
                'batch_r2': self.to_var(data['batch_r2'], self.use_gpu),
                'mode': data['mode']
        })

        return score + weight_meta * score_meta

    def test_one_step_conditional_head_head(self, data, list_info, weight_meta):
        data['batch_r1'], data['batch_R'], data['batch_h2'], data['batch_t2'], data['batch_r2'] = list_info[data['batch_r']].transpose()
        data['batch_h1'] = data['batch_h']
        data['batch_t1'] = data['batch_t']
        data['batch_r'] = data['batch_r1']
        data['batch_t1'] = np.tile(data['batch_t1'], len(data['batch_h1']))
        data['batch_r1'] = np.tile(data['batch_r1'], len(data['batch_h1']))
        
        score = self.model.predict({
                'batch_h': self.to_var(data['batch_h'], self.use_gpu),
                'batch_t': self.to_var(data['batch_t'], self.use_gpu),
                'batch_r': self.to_var(data['batch_r'], self.use_gpu),
                'mode': data['mode'],
        })

        score_meta = self.model.predict_meta({
                'batch_h1': self.to_var(data['batch_h1'], self.use_gpu),
                'batch_t1': self.to_var(data['batch_t1'], self.use_gpu),
                'batch_r1': self.to_var(data['batch_r1'], self.use_gpu),
                'batch_R': self.to_var(data['batch_R'], self.use_gpu),
                'batch_h2': self.to_var(data['batch_h2'], self.use_gpu),
                'batch_t2': self.to_var(data['batch_t2'], self.use_gpu),
                'batch_r2': self.to_var(data['batch_r2'], self.use_gpu),
                'mode': data['mode'],
        })

        return score + weight_meta * score_meta

    def test_one_step_conditional_tail_head(self, data, list_info, weight_meta):
        data['batch_r1'], data['batch_R'], data['batch_h2'], data['batch_t2'], data['batch_r2'] = list_info[data['batch_r']].transpose()
        data['batch_h1'] = data['batch_h']
        data['batch_t1'] = data['batch_t']
        data['batch_r'] = data['batch_r1']
        data['batch_h1'] = np.tile(data['batch_h1'], len(data['batch_t1']))
        data['batch_r1'] = np.tile(data['batch_r1'], len(data['batch_t1']))
        
        score = self.model.predict({
                'batch_h': self.to_var(data['batch_h'], self.use_gpu),
                'batch_t': self.to_var(data['batch_t'], self.use_gpu),
                'batch_r': self.to_var(data['batch_r'], self.use_gpu),
                'mode': data['mode'],
        })

        score_meta = self.model.predict_meta({
                'batch_h1': self.to_var(data['batch_h1'], self.use_gpu),
                'batch_t1': self.to_var(data['batch_t1'], self.use_gpu),
                'batch_r1': self.to_var(data['batch_r1'], self.use_gpu),
                'batch_R': self.to_var(data['batch_R'], self.use_gpu),
                'batch_h2': self.to_var(data['batch_h2'], self.use_gpu),
                'batch_t2': self.to_var(data['batch_t2'], self.use_gpu),
                'batch_r2': self.to_var(data['batch_r2'], self.use_gpu),
                'mode': data['mode'],
        })

        return score + weight_meta * score_meta

    def run_conditional_link_prediction(self, type_constrain=False, list_info=None, weight_meta=0.0, save_dir=None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        if save_dir != None:
            f = open(save_dir, 'w')
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            if save_dir != None:
                h2, r2, t2 = data_tail["batch_h"][0], data_head["batch_r"][0], data_head["batch_t"][0]
                h1, t1, r1, R, r2 = list_info[r2].flatten()
            score = self.test_one_step_conditional_head(data_head, list_info, weight_meta)
            if save_dir != None:
                tmp = score.argsort()
                rank = np.where(tmp == h2)[0][0]
                if rank < 3:
                    f.write("{} {} {} {} _ {} {} | {}\n".format(h1, r1, t1, R, r2, t2, h2))
                    f.write("{}\n".format(' '.join(map(str, list(tmp[:rank + 1])))))
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step_conditional_tail(data_tail, list_info, weight_meta)
            if save_dir != None:
                tmp = score.argsort()
                rank = np.where(tmp == t2)[0][0]
                if rank < 3:
                    f.write("{} {} {} {} {} {} _ | {}\n".format(h1, r1, t1, R, h2, r2, t2))
                    f.write("{}\n".format(' '.join(map(str, list(tmp[:rank + 1])))))
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain)
        
        if save_dir != None:
            f.close()

        mr = self.lib.getTestLinkMR(type_constrain)
        mrr = self.lib.getTestLinkMRR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print (hit10)
        return mr, mrr, hit10, hit3, hit1

    def run_conditional_link_prediction_head(self, type_constrain=False, list_info=None, weight_meta=0.0, save_dir=None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        if save_dir != None:
            f = open(save_dir, 'w')
        training_range = tqdm(self.data_loader)
        for index, [data_head, data_tail] in enumerate(training_range):
            if save_dir != None:
                h1, r1, t1 = data_tail["batch_h"][0], data_head["batch_r"][0], data_head["batch_t"][0]
                r1, R, h2, t2, r2 = list_info[r1].flatten()
            score = self.test_one_step_conditional_head_head(data_head, list_info, weight_meta)
            if save_dir != None:
                tmp = score.argsort()
                rank = np.where(tmp == h1)[0][0]
                if rank < 3:
                    f.write("_ {} {} {} {} {} {} | {}\n".format(r1, t1, R, h2, r2, t2, h1))
                    f.write("{}\n".format(' '.join(map(str, list(tmp[:rank + 1])))))
            self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step_conditional_tail_head(data_tail, list_info, weight_meta)
            if save_dir != None:
                tmp = score.argsort()
                rank = np.where(tmp == t1)[0][0]
                if rank < 3:
                    f.write("{} {} _ {} {} {} {} | {}\n".format(h1, r1, R, h2, r2, t2, t1))
                    f.write("{}\n".format(' '.join(map(str, list(tmp[:rank + 1])))))
            self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
        self.lib.test_link_prediction(type_constrain)
        
        if save_dir != None:
            f.close()

        mr = self.lib.getTestLinkMR(type_constrain)
        mrr = self.lib.getTestLinkMRR(type_constrain)
        hit10 = self.lib.getTestLinkHit10(type_constrain)
        hit3 = self.lib.getTestLinkHit3(type_constrain)
        hit1 = self.lib.getTestLinkHit1(type_constrain)
        print (hit10)
        return mr, mrr, hit10, hit3, hit1

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod