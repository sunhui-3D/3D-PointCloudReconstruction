# -*- coding: utf-8 -*-
import logging
import torch
import sys
import os
sys.path.append(r"../metric/emd/")
sys.path.append(r"../metric/chamfer3D/")
from dist_chamfer_3D import chamfer_3DDist
import emd_module as emd_func

class Metrics(object):
    ITEMS = [{
        'name': 'EMD_distance',
        'enabled': True,
        'eval_func': 'cls._get_emd_distance',
        'eval_object': emd_func.emdModule().cuda(),
        'is_greater_better': False,
        'init_value': 32767
        },
    {
        'name': 'ChamferDistance',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distance',
        'eval_object': chamfer_3DDist().cuda(),
        'is_greater_better': False,
        'init_value': 32767
        }]

    @classmethod
    def get(cls, pred, gt):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            _values[i] = eval_func(pred, gt)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_emd_distance(cls, pred, gt):
        emd_distance = cls.ITEMS[0]['eval_object']
        emd_1, _ = emd_distance(pred, gt, eps=0.005, iters=50)
        emd_loss = torch.sqrt(emd_1).mean(1).mean()
        return emd_loss.item()*100

    @classmethod
    def _get_chamfer_distance(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        dist1, dist2, idx1, idx2 = chamfer_distance(pred, gt)
        chamfer_loss = torch.mean(dist1)+torch.mean(dist2)
        return chamfer_loss.item() * 100

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
