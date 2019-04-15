#encoding:utf-8
import numpy as np
import datetime
import os
import requests
import pandas as pd
import re



def get_date_data(n=5000):
    date_cn = []
    date_en = []
    for timestamp in np.random.randint(143835585, 2043835585, n):
        date = datetime.datetime.fromtimestamp(timestamp)
        date_cn.append(date.strftime("%y-%m-%d"))  #33-04-30   16-02-25
        date_en.append(date.strftime("%d/%b/%Y"))  #30/Apr/2033   25/Feb/2016

    #这里vocab的元素 是0-9的数字，Jun-Dec的月份，几个特殊符号
    vocab = set([str(i) for i in range(0, 10)] + ["-", "/", "<GO>", "<EOS>"] + [i.split("/")[1] for i in date_en])
    v2i = {v: i for i, v in enumerate(vocab, start=1)}
    v2i["<PAD>"] = 0
    vocab.add("<PAD>")

    i2v = {i: v for v, i in v2i.items()}

    x, y = [], []
    for cn, en in zip(date_cn, date_en):
        x.append([v2i[v] for v in cn])
        #y的格式是 <GO> 25/ Feb /2016 <EOS>, 分别对这几个取id,即v2i。注意vocab只有数字0-9，25这种要拆开成2和5
        y.append(
            [v2i["<GO>"], ] + [v2i[v] for v in en[:3]] + [v2i[en[3:6]], ] + [v2i[v] for v in en[6:]] + [v2i["<EOS>"], ])
    x, y = np.array(x), np.array(y)  #这一步就是把16-02-25, 25/Feb/2016转换为数字类型

    # print v2i
    # print i2v
    # print x
    # print y
    return vocab, x, y, v2i, i2v, date_cn, date_en


def pad_zero(seqs, max_len): #末尾补0
    padded = np.zeros((len(seqs), max_len), dtype=np.int32)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded
