# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time
import random
import math
import logging
import json
from pathlib import Path
from six.moves import input
from collections import namedtuple
from operator import itemgetter
from yata.fields import Words, Categorical

from dataprep import get_dataset, get_topics
from util import save_snapshot, load_snapshot, load_last_snapshot, \
    open_result, Variable, use_cuda


def train(model, args):
    logging.info('args: %s' % str(args))
    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))
    logging.info('loading dataset')
    data = get_dataset(args.dataset)
    data.random_level = args.random_level

    if args.split_method == 'user':
        data, _ = data.split_user(args.frac)
    elif args.split_method == 'future':
        data, _ = data.split_future(args.frac)
    elif args.split_method == 'old':
        data, _, _, _ = data.split()

    data = data.get_seq()

    if type(model).__name__.startswith('DK'):
        topic_dic = {}
        kcat = Categorical(one_hot=True)
        kcat.load_dict(open('data/know_list.txt').read().split('\n'))
        for line in open('data/id_know.txt'):
            uuid, know = line.strip().split(' ')
            know = know.split(',')
            topic_dic[uuid] = \
                torch.LongTensor(kcat.apply(None, know)) \
                .max(0)[0] \
                .type(torch.LongTensor)
        zero = [0] * len(kcat.apply(None, '<NULL>'))
    else:
        topics = get_topics(args.dataset, model.words)

    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = load_last_snapshot(model, args.workspace)
    if use_cuda:
        model.cuda()

    for epoch in range(start_epoch, args.epochs):
        logging.info(('epoch {}:'.format(epoch)))
        then = time.time()

        total_loss = 0
        total_mae = 0
        total_acc = 0
        total_seq_cnt = 0

        users = list(data)
        random.shuffle(users)
        seq_cnt = len(users)

        MSE = torch.nn.MSELoss()
        MAE = torch.nn.L1Loss()

        for user in users:
            total_seq_cnt += 1

            seq = data[user]
            length = len(seq)

            optimizer.zero_grad()

            loss = 0
            mae = 0
            acc = 0

            h = None

            for i, item in enumerate(seq):
                if type(model).__name__.startswith('DK'):
                    if item.topic in topic_dic:
                        x = topic_dic[item.topic]
                    else:
                        x = zero
                else:
                    x = topics.get(item.topic).content
                x = Variable(torch.LongTensor(x))
                # print(x.size())
                score = Variable(torch.FloatTensor([round(item.score)]))
                t = Variable(torch.FloatTensor([item.time]))
                s, h = model(x, score, t, h)
                if args.loss == 'cross_entropy':
                    loss += F.binary_cross_entropy_with_logits(
                        s, score.view_as(s))
                    m = MAE(F.sigmoid(s), score).data[0]
                else:
                    loss += MSE(s, score)
                    m = MAE(s, score).data[0]
                mae += m
                acc += m < 0.5

            loss /= length
            mae /= length
            acc /= length

            total_loss += loss.data[0]
            total_mae += mae
            total_acc += acc

            loss.backward()
            optimizer.step()

            if total_seq_cnt % args.save_every == 0:
                save_snapshot(model, args.workspace,
                              '%d.%d' % (epoch, total_seq_cnt))

            if total_seq_cnt % args.print_every != 0 and \
                    total_seq_cnt != seq_cnt:
                continue

            now = time.time()
            duration = (now - then) / 60

            logging.info('[%d:%d/%d] (%.2f seqs/min) '
                         'loss %.6f, mae %.6f, acc %.6f' %
                         (epoch, total_seq_cnt, seq_cnt,
                          ((total_seq_cnt - 1) %
                           args.print_every + 1) / duration,
                          total_loss / total_seq_cnt,
                          total_mae / total_seq_cnt,
                          total_acc / total_seq_cnt))
            then = now

        save_snapshot(model, args.workspace, epoch + 1)


def trainn(model, args):
    logging.info('model: %s, setup: %s' % (type(model).__name__, str(model.args)))
    logging.info('loading dataset')
    data = get_dataset(args.dataset)
    data.random_level = args.random_level

    if args.split_method == 'user':
        data, _ = data.split_user(args.frac)
    elif args.split_method == 'future':
        data, _ = data.split_future(args.frac)
    elif args.split_method == 'old':
        data, _, _, _ = data.split()

    data = data.get_seq()

    if args.input_knowledge:
        logging.info('loading knowledge concepts')
        topic_dic = {}
        kcat = Categorical(one_hot=True)
        kcat.load_dict(open(model.args['knows']).read().split('\n'))
        know = 'data/id_firstknow.txt' if 'first' in model.args['knows'] \
            else 'data/id_know.txt'
        for line in open(know):
            uuid, know = line.strip().split(' ')
            know = know.split(',')
            topic_dic[uuid] = torch.LongTensor(kcat.apply(None, know)).max(0)[0]
        zero = [0] * len(kcat.apply(None, '<NULL>'))

    if args.input_text:
        logging.info('loading exercise texts')
        topics = get_topics(args.dataset, model.words)

    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = load_last_snapshot(model, args.workspace)
    if use_cuda:
        model.cuda()

    for epoch in range(start_epoch, args.epochs):
        logging.info('epoch {}:'.format(epoch))
        then = time.time()

        total_loss = 0
        total_mae = 0
        total_acc = 0
        total_seq_cnt = 0

        users = list(data)
        random.shuffle(users)
        seq_cnt = len(users)

        MSE = torch.nn.MSELoss()
        MAE = torch.nn.L1Loss()

        for user in users:
            total_seq_cnt += 1

            seq = data[user]
            seq_length = len(seq)

            optimizer.zero_grad()

            loss = 0
            mae = 0
            acc = 0

            h = None

            for i, item in enumerate(seq):
                # score = round(item.score)
                if args.input_knowledge:
                    if item.topic in topic_dic:
                        knowledge = topic_dic[item.topic]
                    else:
                        knowledge = zero
                    # knowledge = torch.LongTensor(knowledge).view(-1).type(torch.FloatTensor)
                    # one_index = torch.nonzero(knowledge).view(-1)
                    # expand_vec = torch.zeros(knowledge.size()).view(-1)
                    # expand_vec[one_index] = score
                    # cks = torch.cat([knowledge, expand_vec]).view(1, -1)
                    knowledge = Variable(torch.LongTensor(knowledge))
                    # cks = Variable(cks)

                if args.input_text:
                    text = topics.get(item.topic).content
                    text = Variable(torch.LongTensor(text))
                score = Variable(torch.FloatTensor([item.score]))
                item_time = Variable(torch.FloatTensor([item.time]))

                if type(model).__name__.startswith('DK'):
                    s, h = model(knowledge, score, item_time, h)
                elif type(model).__name__.startswith('RA'):
                    s, h = model(text, score, item_time, h)
                elif type(model).__name__.startswith('EK'):
                    s, h = model(text, knowledge, score, item_time, h)

                s = s[0]

                if args.loss == 'cross_entropy':
                    loss += F.binary_cross_entropy_with_logits(s, score.view_as(s))
                    m = MAE(F.sigmoid(s), score).data[0]
                else:
                    loss += MSE(s, score)
                    m = MAE(s, score).data[0]
                mae += m
                acc += m < 0.5

            loss /= seq_length
            mae /= seq_length
            acc = float(acc) / seq_length

            total_loss += loss.data[0]
            total_mae += mae
            total_acc += acc

            loss.backward()
            optimizer.step()

            if total_seq_cnt % args.save_every == 0:
                save_snapshot(model, args.workspace, '%d.%d' % (epoch, total_seq_cnt))

            if total_seq_cnt % args.print_every != 0 and total_seq_cnt != seq_cnt:
                continue

            now = time.time()
            duration = (now - then) / 60

            logging.info('[%d:%d/%d] (%.2f seqs/min) loss %.6f, mae %.6f, acc %.6f' %
                         (epoch,total_seq_cnt, seq_cnt, ((total_seq_cnt-1) % args.print_every + 1)/duration,
                          total_loss/total_seq_cnt, total_mae/total_seq_cnt, total_acc/total_seq_cnt))
            then = now

        save_snapshot(model, args.workspace, epoch + 1)


def test(model, args):
    try:
        torch.set_grad_enabled(False)
    except AttributeError:
        pass
    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))
    logging.info('loading dataset')
    data = get_dataset(args.dataset)
    data.random_level = args.random_level

    if not args.dataset.endswith('test'):
        if args.split_method == 'user':
            _, data = data.split_user(args.frac)
            testsets = [('user_split', data, {})]
        elif args.split_method == 'future':
            _, data = data.split_future(args.frac)
            testsets = [('future_split', data, {})]
        elif args.split_method == 'old':
            trainset, _, _, _ = data.split()
            data = trainset.get_seq()
            train, user, exam, new = data.split()
            train = train.get_seq()
            user = user.get_seq()
            exam = exam.get_seq()
            new = new.get_seq()
            testsets = zip(['user', 'exam', 'new'], [user, exam, new],
                           [{}, train, user])
        else:
            if args.ref_set:
                ref = get_dataset(args.ref_set)
                ref.random_level = args.random_level
                testsets = [(args.dataset.split('/')[-1],
                             data.get_seq(), ref.get_seq())]
            else:
                testsets = [('student', data.get_seq(), {})]
    else:
        testsets = [('school', data.get_seq(), {})]

    if type(model).__name__.startswith('DK'):
        topic_dic = {}
        kcat = Categorical(one_hot=True)
        kcat.load_dict(open('data/know_list.txt').read().split('\n'))
        for line in open('data/id_know.txt'):
            uuid, know = line.strip().split(' ')
            know = know.split(',')
            topic_dic[uuid] = \
                torch.LongTensor(kcat.apply(None, know)) \
                .max(0)[0] \
                .type(torch.LongTensor)
        zero = [0] * len(kcat.apply(None, '<NULL>'))
    else:
        topics = get_topics(args.dataset, model.words)

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
    logging.info('loaded model at epoch %s', str(epoch))

    if use_cuda:
        model.cuda()

    for testset, data, ref_data in testsets:
        logging.info('testing on: %s', testset)
        f = open_result(args.workspace, testset, epoch)

        then = time.time()

        total_mse = 0
        total_mae = 0
        total_acc = 0
        total_seq_cnt = 0

        users = list(data)
        random.shuffle(users)
        seq_cnt = len(users)

        MSE = torch.nn.MSELoss()
        MAE = torch.nn.L1Loss()

        for user in users[:5000]:
            seq = data[user]
            if user in ref_data:
                ref_seq = ref_data[user]
            else:
                ref_seq = []

            seq2 = []
            seen = set()
            for item in ref_seq:
                if item.topic in seen:
                    continue
                seen.add(item.topic)
                seq2.append(item)
            ref_seq = seq2

            seq2 = []
            for item in seq:
                if item.topic in seen:
                    continue
                seen.add(item.topic)
                seq2.append(item)
            seq = seq2

            ref_len = len(ref_seq)
            seq = ref_seq + seq
            length = len(seq)

            if ref_len < args.ref_len:
                length = length + ref_len - args.ref_len
                ref_len = args.ref_len

            if length < 1:
                continue
            total_seq_cnt += 1

            mse = 0
            mae = 0
            acc = 0

            pred_scores = Variable(torch.zeros(len(seq)))

            s = None
            h = None

            for i, item in enumerate(seq):
                if args.test_on_last:
                    x = topics.get(seq[-1].topic).content
                    x = Variable(torch.LongTensor(x), volatile=True)
                    score = Variable(torch.FloatTensor([round(seq[-1].score)]),
                                     volatile=True)
                    t = Variable(torch.FloatTensor([seq[-1].time]),
                                 volatile=True)
                    s, _ = model(x, score, t, h)
                    s_last = torch.clamp(s, 0, 1)
                if type(model).__name__.startswith('DK'):
                    if item.topic in topic_dic:
                        x = topic_dic[item.topic]
                    else:
                        x = zero
                else:
                    x = topics.get(item.topic).content
                x = Variable(torch.LongTensor(x))
                score = Variable(torch.FloatTensor([round(item.score)]),
                                 volatile=True)
                t = Variable(torch.FloatTensor([item.time]), volatile=True)
                if args.test_as_seq and i > ref_len and ref_len > 0:
                    s, h = model(x, s.view(1), t, h)
                else:
                    if ref_len > 0 and i > ref_len and not args.test_on_one:
                        s, _ = model(x, score, t, h)
                    else:
                        s, h = model(x, score, t, h)
                if args.loss == 'cross_entropy':
                    s = F.sigmoid(s)
                else:
                    s = torch.clamp(s, 0, 1)
                if args.test_on_last:
                    pred_scores[i] = s_last
                else:
                    pred_scores[i] = s
                if i < ref_len:
                    continue
                mse += MSE(s, score)
                m = MAE(s, score).data[0]
                mae += m
                acc += m < 0.5

            print_seq(seq, pred_scores.data.cpu().numpy(), ref_len, f,
                      args.test_on_last)

            mse /= length
            mae /= length
            acc /= length

            total_mse += mse.data[0]
            total_mae += mae
            total_acc += acc

            if total_seq_cnt % args.print_every != 0 and \
                    total_seq_cnt != seq_cnt:
                continue

            now = time.time()
            duration = (now - then) / 60

            logging.info('[%d/%d] (%.2f seqs/min) '
                         'rmse %.6f, mae %.6f, acc %.6f' %
                         (total_seq_cnt, seq_cnt,
                          ((total_seq_cnt - 1) %
                           args.print_every + 1) / duration,
                          math.sqrt(total_mse / total_seq_cnt),
                          total_mae / total_seq_cnt,
                          total_acc / total_seq_cnt))
            then = now

        f.close()


def testfuture(model, args):
    try:
        torch.set_grad_enabled(False)
    except AttributeError:
        pass
    logging.info('model: %s, setup: %s' % (type(model).__name__, str(model.args)))
    logging.info('loading dataset')

    data = get_dataset(args.dataset)
    data.random_level = args.random_level

    if not args.dataset.endswith('test'):
        if args.split_method == 'user':
            _, data = data.split_user(args.frac)
            testsets = [('user_split', data, {})]
        elif args.split_method == 'future':
            _, data = data.split_future(args.frac)
            testsets = [('future_split', data, {})]
        elif args.split_method == 'old':
            trainset, _, _, _ = data.split()
            data = trainset.get_seq()
            train, user, exam, new = data.split()
            train = train.get_seq()
            user = user.get_seq()
            exam = exam.get_seq()
            new = new.get_seq()
            testsets = zip(['user', 'exam', 'new'], [user, exam, new],
                           [{}, train, user])
        else:
            if args.ref_set:
                ref = get_dataset(args.ref_set)
                ref.random_level = args.random_level
                testsets = [(args.dataset.split('/')[-1],
                             data.get_seq(), ref.get_seq())]
            else:
                testsets = [('student', data.get_seq(), {})]
    else:
        testsets = [('school', data.get_seq(), {})]

    if args.input_knowledge:
        logging.info('loading knowledge concepts')
        topic_dic = {}
        kcat = Categorical(one_hot=True)
        kcat.load_dict(open(model.args['knows']).read().split('\n'))
        know = 'data/id_firstknow.txt' if 'first' in model.args['knows'] \
            else 'data/id_know.txt'
        for line in open(know):
            uuid, know = line.strip().split(' ')
            know = know.split(',')
            topic_dic[uuid] = torch.LongTensor(kcat.apply(None, know)).max(0)[0]
        zero = [0] * len(kcat.apply(None, '<NULL>'))

    if args.input_text:
        logging.info('loading exercise texts')
        topics = get_topics(args.dataset, model.words)

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
    logging.info('loaded model at epoch %s', str(epoch))

    if use_cuda:
        model.cuda()

    for testset, data, ref_data in testsets:
        logging.info('testing on: %s', testset)
        f = open_result(args.workspace, testset, epoch)

        then = time.time()

        total_mse = 0
        total_mae = 0
        total_acc = 0
        total_seq_cnt = 0

        users = list(data)
        random.shuffle(users)
        seq_cnt = len(users)

        MSE = torch.nn.MSELoss()
        MAE = torch.nn.L1Loss()

        for user in users[:5000]:
            seq = data[user]
            if user in ref_data:
                ref_seq = ref_data[user]
            else:
                ref_seq = []

            # seq2 = []
            # seen = set()
            # for item in ref_seq:
            #     if item.topic in seen:
            #         continue
            #     seen.add(item.topic)
            #     seq2.append(item)
            # ref_seq = seq2

            # seq2 = []
            # for item in seq:
            #     if item.topic in seen:
            #         continue
            #     seen.add(item.topic)
            #     seq2.append(item)
            # seq = seq2

            ref_len = len(ref_seq)
            seq = ref_seq + seq
            length = len(seq)

            if ref_len < args.ref_len:
                length = length + ref_len - args.ref_len
                ref_len = args.ref_len

            if length < 1:
                continue

            length -= ref_len

            mse = 0
            mae = 0
            acc = 0

            pred_scores = Variable(torch.zeros(len(seq)))

            s = None
            h = None

            for i, item in enumerate(seq):
                if args.input_knowledge:
                    if item.topic in topic_dic:
                        knowledge = topic_dic[item.topic]
                    else:
                        knowledge = zero
                    knowledge = Variable(torch.LongTensor(knowledge))

                if args.input_text:
                    text = topics.get(item.topic).content
                    text = Variable(torch.LongTensor(text))

                score = Variable(torch.FloatTensor([item.score]), volatile=True)
                item_time = Variable(torch.FloatTensor([item.time]), volatile=True)

                # change student state h until the fit process reaches trainset
                # predict on one
                if ref_len > 0 and i > ref_len:
                    if type(model).__name__.startswith('DK'):
                        s, _ = model(knowledge, score, item_time, h)
                    elif type(model).__name__.startswith('RA'):
                        s, _ = model(text, score, item_time, h)
                    elif type(model).__name__.startswith('EK'):
                        s, _ = model(text, knowledge, score, item_time, h)
                else:
                    if type(model).__name__.startswith('DK'):
                        s, h = model(knowledge, score, item_time, h)
                    elif type(model).__name__.startswith('RA'):
                        s, h = model(text, score, item_time, h)
                    elif type(model).__name__.startswith('EK'):
                        s, h = model(text, knowledge, score, item_time, h)

                pred_scores[i] = s

                if args.loss == 'cross_entropy':
                    s = F.sigmoid(s)
                else:
                    s = torch.clamp(s, 0, 1)

                # ignore the result if the fit process is not enough
                if i < ref_len:
                    continue

                mse += MSE(s, score).data[0]
                m = MAE(s, score).data[0]
                mae += m
                acc += m < 0.5

            print_seq(seq, pred_scores.data.cpu().numpy(), ref_len, f, False)
            mse /= length
            mae /= length
            acc = float(acc) / length

            total_mse += mse
            total_mae += mae
            total_acc += acc

            total_seq_cnt += 1

            if total_seq_cnt % args.print_every != 0 and total_seq_cnt != seq_cnt:
                continue

            now = time.time()
            duration = (now - then) / 60

            logging.info('[%d/%d] (%.2f seqs/min) '
                         'rmse %.6f, mae %.6f, acc %.6f' %
                         (total_seq_cnt, seq_cnt,
                          ((total_seq_cnt - 1) %
                           args.print_every + 1) / duration,
                          math.sqrt(total_mse / total_seq_cnt),
                          total_mae / total_seq_cnt,
                          total_acc / total_seq_cnt))
            then = now
        f.close()


def test_future_on_seq(model, args):
    try:
        torch.set_grad_enabled(False)
    except AttributeError:
        pass
    logging.info('model: %s, setup: %s' % (type(model).__name__, str(model.args)))
    logging.info('loading dataset')

    data = get_dataset(args.dataset)
    data.random_level = args.random_level

    if not args.dataset.endswith('test'):
        if args.split_method == 'user':
            _, data = data.split_user(args.frac)
            testsets = [('user_split', data, {})]
        elif args.split_method == 'future':
            _, data = data.split_future(args.frac)
            testsets = [('future_split', data, {})]
        elif args.split_method == 'old':
            trainset, _, _, _ = data.split()
            data = trainset.get_seq()
            train, user, exam, new = data.split()
            train = train.get_seq()
            user = user.get_seq()
            exam = exam.get_seq()
            new = new.get_seq()
            testsets = zip(['user', 'exam', 'new'], [user, exam, new],
                           [{}, train, user])
        else:
            if args.ref_set:
                ref = get_dataset(args.ref_set)
                ref.random_level = args.random_level
                testsets = [(args.dataset.split('/')[-1],
                             data.get_seq(), ref.get_seq())]
            else:
                testsets = [('student', data.get_seq(), {})]
    else:
        testsets = [('school', data.get_seq(), {})]

    if args.input_knowledge:
        logging.info('loading knowledge concepts')
        topic_dic = {}
        kcat = Categorical(one_hot=True)
        kcat.load_dict(open(model.args['knows']).read().split('\n'))
        know = 'data/id_firstknow.txt' if 'first' in model.args['knows'] \
            else 'data/id_know.txt'
        for line in open(know):
            uuid, know = line.strip().split(' ')
            know = know.split(',')
            topic_dic[uuid] = torch.LongTensor(kcat.apply(None, know)).max(0)[0]
        zero = [0] * len(kcat.apply(None, '<NULL>'))

    if args.input_text:
        logging.info('loading exercise texts')
        topics = get_topics(args.dataset, model.words)

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
    logging.info('loaded model at epoch %s', str(epoch))

    if use_cuda:
        model.cuda()

    for testset, data, ref_data in testsets:
        logging.info('testing on: %s', testset)
        f = open_result(args.workspace, testset, epoch)

        then = time.time()

        total_mse = 0
        total_mae = 0
        total_acc = 0
        total_seq_cnt = 0

        users = list(data)
        random.shuffle(users)
        seq_cnt = len(users)

        MSE = torch.nn.MSELoss()
        MAE = torch.nn.L1Loss()

        for user in users[:5000]:
            total_seq_cnt += 1

            seq = data[user]
            if user in ref_data:
                ref_seq = ref_data[user]
            else:
                ref_seq = []

            length = len(seq)
            ref_len = len(ref_seq)
            seq = ref_seq + seq

            if ref_len < args.ref_len:
                length = length + ref_len - args.ref_len
                ref_len = args.ref_len

            if length < 1:
                ref_len = ref_len + length - 1
                length = 1

            mse = 0
            mae = 0
            acc = 0

            seq2 = []
            seen = set()
            for item in seq:
                if item.topic in seen:
                    continue
                seen.add(item.topic)
                seq2.append(item)

            seq = seq2
            length = len(seq) - ref_len

            pred_scores = Variable(torch.zeros(len(seq)))

            s = None
            h = None

            for i, item in enumerate(seq):
                if args.input_knowledge:
                    if item.topic in topic_dic:
                        knowledge = topic_dic[item.topic]
                    else:
                        knowledge = zero
                    knowledge = Variable(torch.LongTensor(knowledge))

                if args.input_text:
                    text = topics.get(item.topic).content
                    text = Variable(torch.LongTensor(text))

                score = Variable(torch.FloatTensor([round(item.score)]), volatile=True)
                item_time = Variable(torch.FloatTensor([item.time]), volatile=True)

                # change student state h by true score if the fit process does not reach trainset
                # change student state h by pred score if the fit process reaches trainset
                # predict on seq
                if ref_len > 0 and i > ref_len:
                    if type(model).__name__.startswith('DK'):
                        s, h = model(knowledge, s.view(1), item_time, h)
                    elif type(model).__name__.startswith('RA'):
                        s, h = model(text, s.view(1), item_time, h)
                    elif type(model).__name__.startswith('EK'):
                        s, h = model(text, knowledge, s.view(1), item_time, h)
                else:
                    if type(model).__name__.startswith('DK'):
                        s, h = model(knowledge, score, item_time, h)
                    elif type(model).__name__.startswith('RA'):
                        s, h = model(text, score, item_time, h)
                    elif type(model).__name__.startswith('EK'):
                        s, h = model(text, knowledge, score, item_time, h)

                pred_scores[i] = s

                if args.loss == 'cross_entropy':
                    s = F.sigmoid(s)
                else:
                    s = torch.clamp(s, 0, 1)

                # ignore the result if the fit process is not enough
                if i < ref_len:
                    continue

                mse += MSE(s, score)
                m = MAE(s, score).data[0]
                mae += m
                acc += m < 0.5

            print_seq(seq, pred_scores.data.cpu().numpy(), ref_len, f, args.test_on_last)

            mse /= length
            mae /= length
            acc /= length

            total_mse += mse.data[0]
            total_mae += mae
            total_acc += acc

            if total_seq_cnt % args.print_every != 0 and total_seq_cnt != seq_cnt:
                continue

            now = time.time()
            duration = (now - then) / 60

            logging.info('[%d/%d] (%.2f seqs/min) '
                         'rmse %.6f, mae %.6f, acc %.6f' %
                         (total_seq_cnt, seq_cnt,
                          ((total_seq_cnt - 1) %
                           args.print_every + 1) / duration,
                          math.sqrt(total_mse / total_seq_cnt),
                          total_mae / total_seq_cnt,
                          total_acc / total_seq_cnt))
            then = now
        f.close()

def testseq(model, args):
    try:
        torch.set_grad_enabled(False)
    except AttributeError:
        pass
    logging.info('model: %s, setup: %s' % (type(model).__name__, str(model.args)))
    logging.info('loading dataset')

    data = get_dataset(args.dataset)
    data.random_level = args.random_level

    if not args.dataset.endswith('test'):
        if args.split_method == 'user':
            _, data = data.split_user(args.frac)
            testsets = [('user_split', data, {})]
        elif args.split_method == 'future':
            _, data = data.split_future(args.frac)
            testsets = [('future_split', data, {})]
        elif args.split_method == 'old':
            trainset, _, _, _ = data.split()
            data = trainset.get_seq()
            train, user, exam, new = data.split()
            train = train.get_seq()
            user = user.get_seq()
            exam = exam.get_seq()
            new = new.get_seq()
            testsets = zip(['user', 'exam', 'new'], [user, exam, new],
                           [{}, train, user])
        else:
            if args.ref_set:
                ref = get_dataset(args.ref_set)
                ref.random_level = args.random_level
                testsets = [(args.dataset.split('/')[-1],
                             data.get_seq(), ref.get_seq())]
            else:
                testsets = [('student', data.get_seq(), {})]
    else:
        testsets = [('school', data.get_seq(), {})]

    if args.input_knowledge:
        logging.info('loading knowledge concepts')
        topic_dic = {}
        kcat = Categorical(one_hot=True)
        kcat.load_dict(open(model.args['knows']).read().split('\n'))
        know = 'data/id_firstknow.txt' if 'first' in model.args['knows'] \
            else 'data/id_know.txt'
        for line in open(know):
            uuid, know = line.strip().split(' ')
            know = know.split(',')
            topic_dic[uuid] = torch.LongTensor(kcat.apply(None, know)).max(0)[0]
        zero = [0] * len(kcat.apply(None, '<NULL>'))

    if args.input_text:
        logging.info('loading exercise texts')
        topics = get_topics(args.dataset, model.words)

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
    logging.info('loaded model at epoch %s', str(epoch))

    if use_cuda:
        model.cuda()

    for testset, data, ref_data in testsets:
        logging.info('testing on: %s', testset)
        f = open_result(args.workspace, testset, epoch)

        then = time.time()

        total_mse = 0
        total_mae = 0
        total_acc = 0
        total_seq_cnt = 0

        users = list(data)
        random.shuffle(users)
        seq_cnt = len(users)

        MSE = torch.nn.MSELoss()
        MAE = torch.nn.L1Loss()

        for user in users[:5000]:
            total_seq_cnt += 1

            seq = data[user]
            if user in ref_data:
                ref_seq = ref_data[user]
            else:
                ref_seq = []

            length = len(seq)
            ref_len = len(ref_seq)
            seq = ref_seq + seq

            if ref_len < args.ref_len:
                length = length + ref_len - args.ref_len
                ref_len = args.ref_len

            if length < 1:
                ref_len = ref_len + length - 1
                length = 1

            mse = 0
            mae = 0
            acc = 0

            # seq2 = []
            # seen = set()
            # for item in seq:
            #     if item.topic in seen:
            #         continue
            #     seen.add(item.topic)
            #     seq2.append(item)

            # seq = seq2
            # length = len(seq) - ref_len

            pred_scores = Variable(torch.zeros(len(seq)))

            s = None
            h = None

            for i, item in enumerate(seq):
                # get last record for testing and current record for updating
                if args.input_knowledge:
                    if item.topic in topic_dic:
                        knowledge = topic_dic[item.topic]
                        knowledge_last = topic_dic[seq[-1].topic]
                    else:
                        knowledge = zero
                        knowledge_last = zero
                    knowledge = Variable(torch.LongTensor(knowledge))
                    knowledge_last = Variable(torch.LongTensor(knowledge_last), volatile=True)

                if args.input_text:
                    text = topics.get(item.topic).content
                    text = Variable(torch.LongTensor(text))
                    text_last = topics.get(seq[-1].topic).content
                    text_last = Variable(torch.LongTensor(text_last), volatile=True)

                score = Variable(torch.FloatTensor([item.score]), volatile=True)
                score_last = Variable(torch.FloatTensor([round(seq[-1].score)]), volatile=True)
                item_time = Variable(torch.FloatTensor([item.time]), volatile=True)
                time_last = Variable(torch.FloatTensor([seq[-1].time]), volatile=True)

                # test last score of each seq for seq figure
                if type(model).__name__.startswith('DK'):
                    s, _ = model(knowledge_last, score_last, time_last, h)
                elif type(model).__name__.startswith('RA'):
                    s, _ = model(text_last, score_last, time_last, h)
                elif type(model).__name__.startswith('EK'):
                    s, _ = model(text_last, knowledge_last, score_last, time_last, h)
                s_last = torch.clamp(s, 0, 1)

                # update student state h until the fit process reaches trainset
                if ref_len > 0 and i > ref_len:
                    if type(model).__name__.startswith('DK'):
                        s, _ = model(knowledge, score, item_time, h)
                    elif type(model).__name__.startswith('RA'):
                        s, _ = model(text, score, item_time, h)
                    elif type(model).__name__.startswith('EK'):
                        s, _ = model(text, knowledge, score, item_time, h)
                else:
                    if type(model).__name__.startswith('DK'):
                        s, h = model(knowledge, score, item_time, h)
                    elif type(model).__name__.startswith('RA'):
                        s, h = model(text, score, item_time, h)
                    elif type(model).__name__.startswith('EK'):
                        s, h = model(text, knowledge, score, item_time, h)

                pred_scores[i] = s_last

                if args.loss == 'cross_entropy':
                    s = F.sigmoid(s)
                else:
                    s = torch.clamp(s, 0, 1)
                if i < ref_len:
                    continue

                mse += MSE(s, score)
                m = MAE(s, score).data[0]
                mae += m
                acc += m < 0.5

            print_seq(seq, pred_scores.data.cpu().numpy(), ref_len, f, True)

            mse /= length
            mae /= length
            acc = float(acc) / length

            total_mse += mse.data[0]
            total_mae += mae
            total_acc += acc

            if total_seq_cnt % args.print_every != 0 and total_seq_cnt != seq_cnt:
                continue

            now = time.time()
            duration = (now - then) / 60

            logging.info('[%d/%d] (%.2f seqs/min) '
                         'rmse %.6f, mae %.6f, acc %.6f' %
                         (total_seq_cnt, seq_cnt,
                          ((total_seq_cnt - 1) %
                           args.print_every + 1) / duration,
                          math.sqrt(total_mse / total_seq_cnt),
                          total_mae / total_seq_cnt,
                          total_acc / total_seq_cnt))
            then = now
        f.close()


def print_seq(seq, pred_scores, ref_len, f, last=False):
    for i in range(len(seq)):
        print(seq[i].topic, i, ref_len, seq[-1 if last else i].score,
              pred_scores[i], file=f)
    f.flush()


def stat_overall(f, with_auc, round_score=False, short=False):
    max_cnt = 0
    total_mse = 0
    total_mae = 0
    total_acc = 0
    total_cnt = 0
    sps = []
    sns = []
    seen = set()

    seq_cnt = 0
    new_seq_flat = False
    auc = 0
    for line in f:
        uuid, i, ref_len, true, pred = line.split()

        i = int(i)
        ref_len = int(ref_len)
        true = float(true)
        if round_score:
            true = round(true)
        pred = float(pred)
        pred = np.clip(pred, 0, 1)
        mae = abs(true - pred)
        acc = mae < 0.5

        if i == 0 and len(sps) > 0 and len(sns) > 0:
            seq_cnt += 1
            auc += calc_auc(sps, sns)
            sps = []
            sns = []
            seen = set()

        if uuid in seen:
            continue
        else:
            seen.add(uuid)

        if i < ref_len:
            continue
        if short and i > ref_len + 10:
            continue

        total_cnt += 1
        total_mae += mae
        total_mse += mae * mae
        total_acc += acc

        if true > 0.8:
            sps.append(pred)
        elif true < 0.2:
            sns.append(pred)

    logging.info('mae: %f\trmse:%f\tacc:%f\tauc:%f' %
                 (total_mae / total_cnt,
                  math.sqrt(total_mse / total_cnt),
                  total_acc / total_cnt,
                  auc/seq_cnt))


def stat_seq(f, with_auc, round_score=False, short=False):
    cnt = [0 for _ in range(5000)]
    accs = [0.0 for _ in range(5000)]
    spss = [[] for _ in range(5000)]
    snss = [[] for _ in range(5000)]
    max_cnt = 0
    total_mse = 0
    total_mae = 0
    total_acc = 0
    total_cnt = 0
    sps = []
    sns = []
    for line in f:
        uuid, i, ref_len, true, pred = line.split()
        i = int(i)
        ref_len = int(ref_len)
        true = float(true)
        if round_score:
            true = round(true)
        pred = float(pred)
        pred = np.clip(pred, 0, 1)
        mae = abs(true - pred)
        acc = mae < 0.5
        cnt[i] += 1
        if cnt[i] > max_cnt:
            max_cnt = cnt[i]
        accs[i] += acc

        if i < ref_len:
            continue
        if short and i > ref_len + 10:
            continue

        if true >= 0.5:
            sps.append(pred)
            spss[i].append(pred)
        else:
            sns.append(pred)
            snss[i].append(pred)
        total_cnt += 1
        total_mae += mae
        total_mse += mae * mae
        total_acc += acc

    for i in range(0, 500):
        if cnt[i] < 10:
            break
        print(i, accs[i] / cnt[i], calc_auc(spss[i], snss[i]), sep='\t')

    auc = 0.5
    if with_auc:
        auc = calc_auc(sps, sns)

    logging.info('mae: %f\trmse:%f\tacc:%f\tauc:%f' %
                 (total_mae / total_cnt,
                  math.sqrt(total_mse / total_cnt),
                  total_acc / total_cnt,
                  auc))


# deprecated
def stat(f, with_auc, round_score=False, short=False):
    cnt = [0 for _ in range(5000)]
    accs = [0.0 for _ in range(5000)]
    spss = [[] for _ in range(5000)]
    snss = [[] for _ in range(5000)]
    max_cnt = 0
    total_mse = 0
    total_mae = 0
    total_acc = 0
    total_cnt = 0
    sps = []
    sns = []
    for line in f:
        uuid, i, ref_len, true, pred = line.split()
        i = int(i)
        ref_len = int(ref_len)
        true = float(true)
        if round_score:
            true = round(true)
        pred = float(pred)
        # pred = np.clip(pred, 0, 1)
        mae = abs(true - pred)
        acc = mae < 0.5
        cnt[i] += 1
        if cnt[i] > max_cnt:
            max_cnt = cnt[i]
        accs[i] += acc

        if true >= 0.5:
            sps.append(pred)
            spss[i].append(pred)
        else:
            sns.append(pred)
            snss[i].append(pred)

        if i < ref_len:
            continue
        if short and i > ref_len + 10:
            continue

        total_cnt += 1
        total_mae += mae
        total_mse += mae * mae
        total_acc += acc

    for i in range(0, 500):
        if cnt[i] < 10:
            break
        print(i, accs[i] / cnt[i], calc_auc(spss[i], snss[i]), sep='\t')

    auc = 0.5
    if with_auc:
        auc = calc_auc(sps, sns)

    logging.info('mae: %f\trmse:%f\tacc:%f\tauc:%f' %
                 (total_mae / total_cnt,
                  math.sqrt(total_mse / total_cnt),
                  total_acc / total_cnt,
                  auc))


def calc_auc(sps, sns):
    auc = 0
    cnt = 0
    sns.sort()
    for sp in sps:
        i = np.searchsorted(sns, sp)
        auc += i
        if i < len(sns) and sns[i] - sp < 1e-6:
            auc += 1
        cnt += len(sns)
    if cnt == 0:
        return 1
    auc /= cnt
    return auc


def predict(model, args):
    try:
        torch.set_grad_enabled(False)
    except AttributeError:
        pass
    logging.info('model: %s, setup: %s' %
                 (type(model).__name__, str(model.args)))
    logging.info('loading dataset')

    if args.snapshot is None:
        epoch = load_last_snapshot(model, args.workspace)
    else:
        epoch = args.snapshot
        load_snapshot(model, args.workspace, epoch)
    logging.info('loaded model at epoch %s', str(epoch))

    to_categorical = Categorical('</s>')
    to_categorical.load_dict(model.words)
    trans = to_categorical(Words(':', null='</s>'))

    while True:
        # loop over inputs
        try:
            line = input()
        except EOFError:
            logging.info('bye')
            break

        try:
            obj = json.loads(line, encoding='utf-8')
            ref_seq = obj['ref']
            pred_seq = obj['pred']
        except (json.decoder.JSONDecodeError, KeyError):
            print('[]')
            continue

        h = None
        for i, item in enumerate(ref_seq):
            x = trans.apply(None, item['fea'])
            x = Variable(torch.LongTensor(x), volatile=True)
            score = Variable(torch.FloatTensor([item['t']]),
                             volatile=True)
            t = Variable(torch.FloatTensor([item['s']]), volatile=True)
            _, h = model(x, score, t, h)

        pred_scores = []

        for i, item in enumerate(pred_seq):
            x = trans.apply(None, item['fea'])
            x = Variable(torch.LongTensor(x), volatile=True)
            score = Variable(torch.FloatTensor([0.]),
                             volatile=True)
            t = Variable(torch.FloatTensor([item['t']]), volatile=True)
            s, _ = model(x, score, t, h)
            pred_scores.append(s.cpu().data[0][0])

        print(pred_scores)


if __name__ == '__main__':
    print(calc_auc([1], [1]))
