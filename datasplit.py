# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

from dataprep import Dataset, get_dataset
from collections import defaultdict
from random import sample, random


if __name__ == '__main__':
    data = get_dataset('full')
    sampled = Dataset()
    user_len = defaultdict(int)
    for r in sample(data.records, int(len(data.records) / 1.25)):
        user_len[r.user] += 1
        ll = user_len[r.user]
        sampled._insert(r)
    sampled.save('data/raw_long/full_sampled.json')

    data = sampled
    data.random_level = 2
    # data.load('data/raw_long/full_sampled.json')

    for frac in [0.6, 0.7, 0.8, 0.9]:
        f = int(frac * 100)
        # print('splitting user at rate %.d%%' % f)
        # train, test = data.split_user(frac)
        # print(len(train.records))
        # print('saving')
        # train.save('data/raw_long/user.train.%.d' % f)
        # test.save('data/raw_long/user.test.%.d' % f)

        print('splitting future at rate %.d' % f)
        train, test = data.split_future(frac)
        print('saving')
        train.save('data/raw_long/future.train.%d' % f)
        test.save('data/raw_long/future.test.%d' % f)
