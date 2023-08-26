import os

import torch


class AverageMeters(object):
    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        self.total_num = total_num or {}

    def reset(self):
        for key in self.dic:
            self.dic[key] = 0
            self.total_num[key] = 0

    def update(self, new_dic):
        for key in new_dic:
            if type(new_dic[key]) is not float:
                value = new_dic[key].item()
            else:
                value = new_dic[key]

            if not key in self.dic:
                self.dic[key] = value
                self.total_num[key] = 1
            else:
                self.dic[key] += value
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.5f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


def write_loss(writer, prefix, avg_meters, iteration):
    for key in avg_meters.keys():
        meter = avg_meters[key]
        writer.add_scalar(
            os.path.join(prefix, key), meter, iteration)
