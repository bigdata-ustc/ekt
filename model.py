# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from dataprep import load_embedding
from util import Variable
import json


#####
# 得分预测模型
###

class RNN(nn.Module):
    """
    得分预测的RNN模型
    """
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--emb_file', default='data/emb_50.txt',
                            help='pretrained word embedding')
        parser.add_argument('--topic_size', '-ts', type=int, default=50,
                            help='topic embedding size')
        parser.add_argument('--seq_hidden_size', '-hs', type=int, default=50,
                            help='sequence embedding size')
        parser.add_argument('--score_mode', '-s',
                            choices=['concat', 'double'], default='double',
                            help='way to combine topics and scores')

    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        self.topic_model = TopicRNNModel(wcnt, emb_size, args['topic_size'])
        self.topic_model.load_emb(embs)
        self.seq_model = SeqModel(args['topic_size'], args['seq_hidden_size'],
                                  args['score_mode'])

    def forward(self, topic, score, time, hidden=None):
        # topic rnn part size: (seq_len, bz) -> (bz, topic_size)
        h = self.topic_model.default_hidden(1)
        topic_v, _ = self.topic_model(topic.view(-1, 1), h)

        s, hidden = self.seq_model(topic_v[0], score, hidden)
        return s, hidden


class Attn(nn.Module):
    """
    得分预测的纯attention模型
    """
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--emb_file', default='data/emb_50.txt',
                            help='pretrained word embedding')
        parser.add_argument('--topic_size', '-ts', type=int, default=50,
                            help='topic embedding size')
        parser.add_argument('-k', type=int, default=5,
                            help='use top k similar topics to predict')

    def __init__(self, args):
        super(Attn, self).__init__()
        self.args = args
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        self.topic_model = TopicRNNModel(wcnt, emb_size, args['topic_size'])
        self.topic_model.load_emb(embs)
        self.seq_model = AttnModel(args['topic_size'], args['k'])

    def forward(self, topic, score, time, hidden=None):
        # topic rnn part size: (seq_len, bz) -> (bz, topic_size)
        h = self.topic_model.default_hidden(1)
        topic_v, _ = self.topic_model(topic.view(-1, 1), h)

        s = self.seq_model(topic_v[0], hidden)

        if hidden is None:
            hidden = topic_v, score

        else:
            vs, scores = hidden
            vs = torch.cat([vs, topic_v])
            scores = torch.cat([scores, score])
            hidden = vs, scores

        return s, hidden


class RA(nn.Module):
    """
    得分预测的RNN+attention模型
    """
    @staticmethod
    def add_arguments(parser):
        RNN.add_arguments(parser)
        parser.add_argument('-k', type=int, default=10,
                            help='use top k similar topics to predict')
        parser.add_argument('-w', '--with_last',
                            action='store_true', help='with last h')
        parser.add_argument('-l', '--num_layers', type=int, default=2,
                            help='#topic rnn layers')

    def __init__(self, args):
        super(RA, self).__init__()
        self.args = args
        #print(self.args)
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        self.topic_model = TopicRNNModel(wcnt, emb_size, args['topic_size'],
                                         num_layers=args['num_layers'])
        self.topic_model.load_emb(embs)
        self.seq_model = AttnSeqModel(args['topic_size'],
                                      args['seq_hidden_size'], args['k'],
                                      args['score_mode'], args['with_last'])

    def forward(self, topic, score, time, hidden=None):
        # topic rnn part size: (seq_len, bz) -> (bz, topic_size)
        h = self.topic_model.default_hidden(1)
        topic_v, _ = self.topic_model(topic.view(-1, 1), h)

        s, h = self.seq_model(topic_v[0], score, hidden)
        if hidden is None:
            hidden = h, topic_v, h
        else:
            _, vs, hs = hidden
            vs = torch.cat([vs, topic_v])
            hs = torch.cat([hs, h])
            hidden = h, vs, hs

        return s, hidden


class RADecay(nn.Module):
    """
    得分预测的RNN+attention模型（增加时间衰减）
    """
    @staticmethod
    def add_arguments(parser):
        RNN.add_arguments(parser)
        parser.add_argument('-k', type=int, default=5,
                            help='use top k similar topics to predict')

    def __init__(self, args):
        super(RADecay, self).__init__()
        self.args = args
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        self.topic_model = TopicRNNModel(wcnt, emb_size, args['topic_size'])
        self.topic_model.load_emb(embs)
        self.seq_model = AttnSeqTimeDecayModel(args['topic_size'],
                                               args['seq_hidden_size'],
                                               args['k'],
                                               args['score_mode'])
        self.vs = None
        self.hs = None

    def forward(self, topic, score, time, hidden=None):
        # topic rnn part size: (seq_len, bz) -> (bz, topic_size)
        h = self.topic_model.default_hidden(1)
        topic_v, _ = self.topic_model(topic.view(-1, 1), h)

        s, h = self.seq_model(topic_v[0], score, time, hidden)

        if hidden is None:
            hidden = topic_v, h, time
        else:
            vs, hs, ts = hidden
            vs = torch.cat([vs, topic_v])
            hs = torch.cat([hs, h])
            ts = torch.cat([ts, time])
            hidden = vs, hs, ts

        return s, hidden


class LSTMM(nn.Module):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--emb_file', default='data/emb_50.txt',
                            help='pretrained word embedding')
        parser.add_argument('--seq_hidden_size', '-hs', type=int, default=50,
                            help='sequence embedding size')
        parser.add_argument('--topic_size', '-ts', type=int, default=50,
                            help='topic embedding size')
        parser.add_argument('--score_mode', '-s',
                            choices=['concat', 'double'], default='double',
                            help='way to combine topics and scores')

    def __init__(self, args):
        super(LSTMM, self).__init__()
        self.args = args
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        self.embedding = nn.Embedding(wcnt, emb_size, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embs))
        self.seq_model = SeqModel(args['topic_size'], args['seq_hidden_size'],
                                  args['score_mode'])

    def forward(self, topic, score, time, hidden=None):
        topic_v = self.embedding(topic).mean(0, keepdim=True)
        s, hidden = self.seq_model(topic_v[0], score, hidden)
        return s, hidden


class LSTMA(nn.Module):
    @staticmethod
    def add_arguments(parser):
        LSTMM.add_arguments(parser)
        parser.add_argument('-k', type=int, default=10,
                            help='use top k similar topics to predict')
        parser.add_argument('-w', '--with_last',
                            action='store_true', help='with last h')

    def __init__(self, args):
        super(LSTMA, self).__init__()
        self.args = args
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        self.embedding = nn.Embedding(wcnt, emb_size, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embs))
        self.seq_model = AttnSeqModel(args['topic_size'],
                                      args['seq_hidden_size'],
                                      args['k'],
                                      args['score_mode'],
                                      args['with_last'])
        self.vs = None
        self.hs = None

    def forward(self, topic, score, time, hidden=None):
        topic_v = self.embedding(topic).mean(0, keepdim=True)

        s, h = self.seq_model(topic_v[0], score, hidden)
        if hidden is None:
            hidden = h, topic_v, h
        else:
            _, vs, hs = hidden
            vs = torch.cat([vs, topic_v])
            hs = torch.cat([hs, h])
            hidden = h, vs, hs

        return s, hidden


class DKT(nn.Module):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--tcnt', '-tc', type=int, default=0,
                            help='different topic count')
        parser.add_argument('--score_mode', '-s',
                            choices=['concat', 'double'], default='double',
                            help='way to combine topics and scores')

    def __init__(self, args):
        super(DKT, self).__init__()
        self.args = args
        if args['tcnt'] == 0:
            dic = open('data/know_list.txt').read().split('\n')
            args['tcnt'] = len(dic)
        self.tcnt = args['tcnt']
        self.seq_model = DKTModel(self.tcnt)

    def forward(self, topic, score, time, hidden=None):
        s, hidden = self.seq_model(topic, score, hidden)
        return s, hidden


class DKNM(nn.Module):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--emb_file', default='data/emb_50.txt',
                            help='pretrained word embedding')
        parser.add_argument('--tcnt', '-tc', type=int, default=0,
                            help='different topic count')
        parser.add_argument('--seq_hidden_size', '-hs', type=int, default=50,
                            help='sequence embedding size')
        parser.add_argument('--topic_size', '-ts', type=int, default=50,
                            help='topic embedding size')
        parser.add_argument('--score_mode', '-s',
                            choices=['concat', 'double'], default='double',
                            help='way to combine topics and scores')

    def __init__(self, args):
        super(DKNM, self).__init__()
        self.args = args
        if args['tcnt'] == 0:
            dic = open('data/know_list.txt').read().split('\n')
            args['tcnt'] = len(dic)
        self.tcnt = args['tcnt']
        self.embedding = nn.Linear(self.tcnt, args['topic_size'])
        self.seq_model = SeqModel(args['topic_size'], args['seq_hidden_size'],
                                  args['score_mode'])

    def forward(self, topic, score, time, hidden=None):
        topic_v = self.embedding(topic.type_as(score).view(1, -1))
        s, hidden = self.seq_model(topic_v[0], score, hidden)
        return s, hidden


class DKNA(nn.Module):
    @staticmethod
    def add_arguments(parser):
        DKNM.add_arguments(parser)
        parser.add_argument('-k', type=int, default=10,
                            help='use top k similar topics to predict')
        parser.add_argument('-w', '--with_last',
                            action='store_true', help='with last h')

    def __init__(self, args):
        super(DKNA, self).__init__()
        self.args = args
        if args['tcnt'] == 0:
            dic = open('data/know_list.txt').read().split('\n')
            args['tcnt'] = len(dic)
        self.tcnt = args['tcnt']
        self.embedding = nn.Linear(self.tcnt, args['topic_size'])
        self.seq_model = AttnSeqModel(args['topic_size'],
                                      args['seq_hidden_size'],
                                      args['k'],
                                      args['score_mode'],
                                      args['with_last'])
        self.vs = None
        self.hs = None

    def forward(self, topic, score, time, hidden=None):
        topic_v = self.embedding(topic.type_as(score).view(1, -1))

        s, h = self.seq_model(topic_v[0], score, hidden)
        if hidden is None:
            hidden = h, topic_v, h
        else:
            _, vs, hs = hidden
            vs = torch.cat([vs, topic_v])
            hs = torch.cat([hs, h])
            hidden = h, vs, hs

        return s, hidden


class EKTM(nn.Module):
    """
    Knowledge Tracing Model with Markov property combined with exercise texts and knowledge concepts
    """
    @staticmethod
    def add_arguments(parser):
        RNN.add_arguments(parser)
        parser.add_argument('-k', type=int, default=10, help='use top k similar topics to predict')
        parser.add_argument('-kc', '--kcnt', type=int, default=0, help='numbers of knowledge concepts')
        parser.add_argument('-ks', '--knowledge_hidden_size', type=int, default=25, help='knowledge emb size')
        parser.add_argument('-l', '--num_layers', type=int, default=1, help='#topic rnn layers')
        # parser.add_argument('-hs', '--seq_hidden_size', type=int, default=50, help='student seq emb size')
        # parser.add_argument('-ts', '--topic_size', type=int, default=50, help='exercise emb size')
        # parser.add_argument('-s', '--score_mode', choices=['concat', 'double'], default='double',
        #                     help='way to combine exercise and score')

    def __init__(self, args):
        super(EKTM, self).__init__()
        self.args = args
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        if args['kcnt'] == 0:
            know_dic = open('data/firstknow_list.txt').read().split('\n')
            args['kcnt'] = len(know_dic)
        self.kcnt = args['kcnt']

        # knowledge embedding module
        self.knowledge_model = KnowledgeModel(self.kcnt, args['knowledge_hidden_size'])

        # exercise embedding module
        self.topic_model = TopicRNNModel(wcnt, emb_size, args['topic_size'], num_layers=args['num_layers'])

        self.topic_model.load_emb(embs)

        # student seq module
        self.seq_model = EKTSeqModel(args['topic_size'], args['knowledge_hidden_size'], args['kcnt'],
                                     args['seq_hidden_size'], args['score_mode'])

    def forward(self, topic, knowledge, score, time, hidden=None):
        # print(knowledge.size())
        k = self.knowledge_model(knowledge)
        # print(knowledge.size())

        topic_h = self.topic_model.default_hidden(1)
        topic_v, _ = self.topic_model(topic.view(-1, 1), topic_h)

        s, h = self.seq_model(topic_v[0], k, knowledge, score, hidden)
        return s, h


class EKTA(nn.Module):
    """
    Knowledge Tracing Model with Attention mechnaism combined with exercise texts and knowledge concepts
    """
    @staticmethod
    def add_arguments(parser):
        RNN.add_arguments(parser)
        parser.add_argument('-k', type=int, default=10, help='use top k similar topics to predict')
        parser.add_argument('-kc', '--kcnt', type=int, default=0, help='numbers of knowledge concepts')
        parser.add_argument('-ks', '--knowledge_hidden_size', type=int, default=25, help='knowledge emb size')
        parser.add_argument('-l', '--num_layers', type=int, default=1, help='#topic rnn layers')

    def __init__(self, args):
        super(EKTA, self).__init__()
        self.args = args
        wcnt, emb_size, words, embs = load_embedding(args['emb_file'])
        self.words = words
        if args['kcnt'] == 0:
            know_dic = open('data/firstknow_list.txt').read().split('\n')
            args['kcnt'] = len(know_dic)
        self.kcnt = args['kcnt']

        # knowledge embedding module
        self.knowledge_model = KnowledgeModel(self.kcnt, args['knowledge_hidden_size'])

        # exercise embedding module
        self.topic_model = TopicRNNModel(wcnt, emb_size, args['topic_size'], num_layers=args['num_layers'])

        self.topic_model.load_emb(embs)

        # student seq module
        self.seq_model = EKTAttnSeqModel(args['topic_size'], args['knowledge_hidden_size'], args['kcnt'],
                                         args['seq_hidden_size'], args['k'], args['score_mode'])

    def forward(self, topic, knowledge, score, time, hidden=None, alpha=False):
        # print(knowledge.size())
        k = self.knowledge_model(knowledge)
        # print(knowledge.size())
        topic_h = self.topic_model.default_hidden(1)
        topic_v, _ = self.topic_model(topic.view(-1, 1), topic_h)

        s, h, a = self.seq_model(topic_v[0], k, knowledge, score, hidden)
        if hidden is None:
            hidden = h, topic_v, h
        else:
            _, vs, hs = hidden
            vs = torch.cat([vs, topic_v])
            hs = torch.cat([hs, h])
            hidden = h, vs, hs

        if alpha:
            return s, hidden, a
        else:
            return s, hidden


class DKVMN(nn.Module):
    """
    Dynamic Key-Value Memory Networks for Knowledge Tracing at WWW'2017
    """

    @staticmethod
    def add_arguments(parser):
        RNN.add_arguments(parser)
        parser.add_argument('-k', type=int, default=10, help='use top k similar topics to predict')
        parser.add_argument('--knows', default='data/know_list.txt', help='numbers of knowledge concepts')
        parser.add_argument('-ks', '--knowledge_hidden_size', type=int, default=25, help='knowledge emb size')
        parser.add_argument('-l', '--num_layers', type=int, default=2, help='#topic rnn layers')
        # parser.add_argument('-es', '--erase_vector_size', type=float, default=25, help='erase vector emb size')
        # parser.add_argument('-as', '--add_vector_size', type=float, default=25, help='add vector emb size')

    def __init__(self, args):
        super(DKVMN, self).__init__()
        self.args = args
        know_dic = open(args['knows']).read().split('\n')
        args['kcnt'] = len(know_dic)
        self.kcnt = args['kcnt']
        self.valve_size = args['knowledge_hidden_size'] * 2

        # knowledge embedding module
        self.knowledge_model = KnowledgeModel(self.kcnt, args['knowledge_hidden_size'])
        # student seq module
        self.seq_model = DKVMNSeqModel(args['knowledge_hidden_size'], 30, args['kcnt'], args['seq_hidden_size'],
                                       self.valve_size)

    def forward(self, knowledge, score, time, hidden=None):
        # print(knowledge)
        expand_vec = knowledge.float().view(-1) * score
        # print(expand_vec)
        cks = torch.cat([knowledge.float().view(-1), expand_vec]).view(1, -1)
        # print(cks)

        knowledge = self.knowledge_model(knowledge)

        s, h = self.seq_model(cks, knowledge, score, hidden)
        return s, h


#######
# knowledge Representation module
#######
class KnowledgeModel(nn.Module):
    """
    Transform Knowledge index to knowledge embedding
    """

    def __init__(self, know_len, know_emb_size):
        super(KnowledgeModel, self).__init__()
        self.knowledge_embedding = nn.Linear(know_len, know_emb_size)

    def forward(self, knowledge):
        return self.knowledge_embedding(knowledge.float().view(1, -1))


class DKVMNSeqModel(nn.Module):
    """
    DKVMN seq model
    """

    def __init__(self, know_emb_size, know_length, kcnt, seq_hidden_size, value_size):
        super(DKVMNSeqModel, self).__init__()
        self.know_emb_size = know_emb_size
        self.know_length = know_length
        self.seq_hidden_size = seq_hidden_size
        # self.erase_size = erase_size
        # self.add_size = add_size
        self.value_size = value_size

        # knowledge memory matrix
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1)

        # read process embedding module
        self.ft_embedding = nn.Linear(self.seq_hidden_size + self.know_emb_size, 50)
        self.score_layer = nn.Linear(50, 1)

        # write process embedding module
        # erase_size = add_size = seq_hidden_size
        self.cks_embedding = nn.Linear(kcnt * 2, self.value_size)
        self.erase_embedding = nn.Linear(self.value_size, self.seq_hidden_size)
        self.add_embedding = nn.Linear(self.value_size, self.seq_hidden_size)

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(know_length, seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)

    def forward(self, cks, kn, s, h):
        if h is None:
            h = self.h_initial.view(self.know_length * self.seq_hidden_size)

        # calculate alpha weights of knowledges using dot product
        alpha = torch.mm(self.knowledge_memory, kn.view(-1, 1)).view(-1)
        alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

        # read process
        rt = torch.mm(alpha, h.view(self.know_length, self.seq_hidden_size)).view(-1)
        com_r_k = torch.cat([rt, kn.view(-1)]).view(1, -1)
        # print(com_r_k.size())
        ft = torch.tanh(self.ft_embedding(com_r_k))
        predict_score = torch.sigmoid(self.score_layer(ft))

        # write process
        vt = self.cks_embedding(cks)
        et = torch.sigmoid(self.erase_embedding(vt))
        at = torch.tanh(self.add_embedding(vt))
        ht = h * (1 - (alpha.view(-1, 1) * et).view(-1))
        h = ht + (alpha.view(-1, 1) * at).view(-1)
        return predict_score.view(1), h


class EKTSeqModel(nn.Module):
    """
    Student seq modeling combined with exercise texts and knowledge point
    """

    def __init__(self, topic_size, know_emb_size, know_length, seq_hidden_size, score_mode, num_layers=1):
        super(EKTSeqModel, self).__init__()
        self.topic_size = topic_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length
        self.score_mode = score_mode
        self.num_layers = num_layers
        # self.with_last = with_last

        # Knowledge memory matrix
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1)

        # Student seq rnn
        if self.score_mode == 'concat':
            self.rnn = nn.GRU(self.topic_size + 1, seq_hidden_size, num_layers)
        else:
            self.rnn = nn.GRU(self.topic_size * 2 + 1, seq_hidden_size, num_layers)

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(know_length, seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)

        # prediction layer
        self.score_layer = nn.Linear(topic_size + seq_hidden_size, 1)

    def forward(self, v, kn, ko, s, h, beta=None):
        if h is None:
             h = self.h_initial.view(self.num_layers, self.know_length, self.seq_hidden_size)
             length = Variable(torch.FloatTensor([0.]))

        # calculate alpha weights of knowledges using dot product
        # print(self.knowledge_memory.size())
        # print(kn.view(-1, 1))
        if beta is None:
            alpha = torch.mm(self.knowledge_memory, kn.view(-1, 1)).view(-1)
            beta = nn.functional.softmax(alpha.view(1, -1), dim=-1)
            # print(beta.argmax(1))

        # print(alpha.size())

        # print(h.view(self.know_length, self.seq_hidden_size).size())
        # print(h.type())
        # predict score at time t
        hkp = torch.mm(beta, h.view(self.know_length, self.seq_hidden_size)).view(-1)
        # print(hkp.size())
        pred_v = torch.cat([v, hkp]).view(1, -1)
        # print(pred_v.size())
        predict_score = self.score_layer(pred_v)

        # seq states update
        if self.score_mode == 'concat':
            x = v
        else:
            x = torch.cat([v * (s >= 0.5).type_as(v).expand_as(v),
                           v * (s < 0.5).type_as(v).expand_as(v)])
        x = torch.cat([x, s])

        # print(x.size())
        # print(torch.ones(self.know_length,1).size())
        # print(x.view(1, -1).size())
        # print(x.type())
        # xk = torch.mm(torch.ones(self.know_length, 1), x.view(1, -1))
        xk = x.view(1, -1).expand(self.know_length, -1)
        xk = beta.view(-1, 1) * xk
        # xk = ko.float().view(-1, 1) * xk
        # print(xk.size())
        # print(alpha.size())
        # xk = torch.mm(alpha, xk).view(-1)
        # thresh, idx = alpha.topk(5)
        # alpha = (alpha >= thresh[0, 4]).float()
        # xk = alpha.view(-1, 1) * xk
        # xk = Variable(torch.zeros_like(x)).expand(self.know_length, -1)

        _, h = self.rnn(xk.unsqueeze(0), h)
        return predict_score.view(1), h


class EKTAttnSeqModel(nn.Module):
    """
    Student seq modeling combined with exercise texts and knowledge point
    """

    def __init__(self, topic_size, know_emb_size, know_length, seq_hidden_size, k, score_mode, num_layers=1):
        super(EKTAttnSeqModel, self).__init__()
        self.topic_size = topic_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length
        self.score_mode = score_mode
        self.num_layers = num_layers
        self.k = k
        # self.with_last = with_last

        # Knowledge memory matrix
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1)

        # Student seq rnn
        if self.score_mode == 'concat':
            self.rnn = nn.GRU(self.topic_size + 1, seq_hidden_size, num_layers)
        else:
            self.rnn = nn.GRU(self.topic_size * 2 + 1, seq_hidden_size, num_layers)

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(know_length, seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)

        # prediction layer
        self.score_layer = nn.Linear(topic_size + seq_hidden_size, 1)
        self.k = k

    def forward(self, v, kn, ko, s, hidden):
        if hidden is None:
            h = self.h_initial.view(self.num_layers, self.know_length, self.seq_hidden_size)
            attn_h = self.h_initial
            length = Variable(torch.FloatTensor([0.]))
            beta = None

        else:

            h, vs, hs = hidden

            # calculate beta weights of seqs using dot product
            beta = torch.mm(vs, v.view(-1, 1)).view(-1)
            beta, idx = beta.topk(min(len(beta), self.k), sorted=False)
            beta = nn.functional.softmax(beta.view(1, -1), dim=-1)
            length = Variable(torch.FloatTensor([beta.size()[1]]))

            hs = hs.view(-1, self.know_length * self.seq_hidden_size)
            attn_h = torch.mm(beta, torch.index_select(hs, 0, idx)).view(-1)

        # calculate alpha weights of knowledges using dot product
        alpha = torch.mm(self.knowledge_memory, kn.view(-1, 1)).view(-1)
        alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

        hkp = torch.mm(alpha, attn_h.view(self.know_length, self.seq_hidden_size)).view(-1)
        pred_v = torch.cat([v, hkp]).view(1, -1)
        predict_score = self.score_layer(pred_v)

        # seq states update
        if self.score_mode == 'concat':
            x = v
        else:
            x = torch.cat([v * (s >= 0.5).type_as(v).expand_as(v), v * (s < 0.5).type_as(v).expand_as(v)])
        x = torch.cat([x, s])

        # print(x.size())
        # print(torch.ones(self.know_length,1).size())
        # print(x.view(1, -1).size())
        # print(x.type())
        # xk = torch.mm(torch.ones(self.know_length, 1), x.view(1, -1))
        xk = x.view(1, -1).expand(self.know_length, -1)
        xk = alpha.view(-1, 1) * xk
        # xk = ko.float().view(-1, 1) * xk
        # xk = torch.mm(alpha, xk).view(-1)

        _, h = self.rnn(xk.unsqueeze(0), h)
        return predict_score.view(1), h, beta


#####
# 题目表示、序列表示等模块
###
class TopicIdModel(nn.Module):
    """
    对题号embedding
    """

    def __init__(self, wcnt, word_emb_size):
        super(TopicIdModel, self).__init__()
        self.embedding = nn.Embedding(wcnt, word_emb_size, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)[0]


class TopicRNNModel(nn.Module):
    """
    双向RNN（GRU）建模题面
    """

    def __init__(self, wcnt, emb_size, topic_size, num_layers=2):
        super(TopicRNNModel, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(wcnt, emb_size, padding_idx=0)
        if num_layers > 1:
            self.emb_size = topic_size
            self.rnn = nn.GRU(emb_size, topic_size, 1,
                              bidirectional=True,
                              dropout=0.1)
            self.output = nn.GRU(topic_size * 2,
                                 topic_size, num_layers - 1,
                                 dropout=0.1)
        else:
            self.emb_size = topic_size // 2
            self.rnn = nn.GRU(emb_size, topic_size // 2, 1,
                              bidirectional=True)

    def forward(self, input, hidden):
        x = self.embedding(input)
        # print(x.size())
        # exit(0)
        y, h1 = self.rnn(x, hidden[0])
        if self.num_layers > 1:
            y, h2 = self.output(y, hidden[1])
            return y[-1], (h1, h2)
        else:
            y, _ = torch.max(y, 0)
            return y, (h1, None)

    def default_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.emb_size)), \
            Variable(torch.zeros(self.num_layers - 1,
                                 batch_size, self.emb_size)) \
            if self.num_layers > 1 else None

    def load_emb(self, emb):
        self.embedding.weight.data.copy_(torch.from_numpy(emb))


class DKTModel(nn.Module):
    """
    做题记录序列的RNN（GRU）单元
    """

    def __init__(self, topic_size):
        super(DKTModel, self).__init__()
        self.topic_size = topic_size
        self.rnn = nn.GRU(topic_size * 2, topic_size, 1)
        self.score = nn.Linear(topic_size * 2, 1)

    def forward(self, v, s, h):
        if h is None:
            h = self.default_hidden()

        v = v.type_as(h)
        score = self.score(torch.cat([h.view(-1), v.view(-1)]))

        x = torch.cat([v.view(-1),
                       (v * (s > 0.5).type_as(v).
                        expand_as(v).type_as(v)).view(-1)])
        _, h = self.rnn(x.view(1, 1, -1), h)
        return score.view(1), h

    def default_hidden(self):
        return Variable(torch.zeros(1, 1, self.topic_size))


class SeqModel(nn.Module):
    """
    做题记录序列的RNN（GRU）单元
    """

    def __init__(self, topic_size, seq_hidden_size, score_mode, num_layers=1):
        super(SeqModel, self).__init__()
        self.topic_size = seq_hidden_size
        self.seq_hidden_size = topic_size
        self.num_layers = num_layers
        self.score_mode = score_mode
        if self.score_mode == 'concat':
            self.rnn = nn.GRU(topic_size + 1, seq_hidden_size, num_layers)
        else:
            self.rnn = nn.GRU(topic_size * 2 + 1, seq_hidden_size, num_layers)
        self.score = nn.Linear(seq_hidden_size + topic_size, 1)

    def forward(self, v, s, h):
        if h is None:
            h = self.default_hidden()
        pred_v = torch.cat([v, h.view(-1)])
        score = self.score(pred_v.view(1, -1))

        if self.score_mode == 'concat':
            x = v
        else:
            x = torch.cat([v * (s >= 0.5).type_as(v).expand_as(v),
                           v * (s < 0.5).type_as(v).expand_as(v)])
        x = torch.cat([x, s])

        _, h = self.rnn(x.view(1, 1, -1), h)
        return score.view(1), h

    def default_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.seq_hidden_size))


class AttnModel(nn.Module):
    """
    做题记录序列的纯attention模型单元（alpha：题面embedding点乘）
    """

    def __init__(self, topic_size, k):
        super(AttnModel, self).__init__()
        self.user_emb_size = topic_size
        self.k = k
        self.initial_guess = Variable(torch.zeros(1), requires_grad=True)

    def forward(self, v, h):
        if h is None:
            return self.initial_guess
        else:
            vs, scores = h
            scores = scores.view(-1, 1)

            # calculate alpha using dot product
            alpha = torch.mm(vs, v.view(-1, 1)).view(-1)
            alpha, idx = alpha.topk(min(len(alpha), self.k), sorted=False)
            alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

            score = torch.mm(alpha, torch.index_select(scores, 0, idx))
            return score.view(1, 1)


class AttnSeqModel(nn.Module):
    """
    做题记录序列的RNN+attention模型单元（alpha：题面embedding点乘）
    """

    def __init__(self, topic_size, seq_hidden_size, k,
                 score_mode, with_last, num_layers=1):
        super(AttnSeqModel, self).__init__()
        self.topic_size = topic_size
        self.seq_hidden_size = seq_hidden_size
        self.num_layers = num_layers
        self.score_mode = score_mode
        if self.score_mode == 'concat':
            self.rnn = nn.GRU(topic_size + 1, seq_hidden_size, num_layers)
        else:
            self.rnn = nn.GRU(topic_size * 2 + 1, seq_hidden_size, num_layers)
        self.with_last = with_last
        h_size = seq_hidden_size * 2 + 1 if with_last else seq_hidden_size
        self.score = nn.Linear(topic_size + h_size, 1)
        self.initial_h = nn.Parameter(torch.zeros(self.num_layers *
                                                  self.seq_hidden_size))
        self.initial_h.data.uniform_(-1., 1.)
        self.k = k

    def forward(self, v, s, hidden):
        if hidden is None:
            h = self.initial_h.view(self.num_layers, 1, self.seq_hidden_size)
            attn_h = self.initial_h
            length = Variable(torch.FloatTensor([0.]))
        else:
            h, vs, hs = hidden
            # print(h)
            # print('start')
            # print(vs.size())
            # print(v.size())
            # print(v.view(-1,1).size())
            # print(torch.mm(vs,v.view(-1,1)).size())

            # print(hs)

            # calculate alpha using dot product
            alpha = torch.mm(vs, v.view(-1, 1)).view(-1)
            # print(alpha.size())
            # print('end')
            # print(alpha.size())
            alpha, idx = alpha.topk(min(len(alpha), self.k), sorted=False)
            alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

            length = Variable(torch.FloatTensor([alpha.size()[1]]))

            # flatten each h
            hs = hs.view(-1, self.num_layers * self.seq_hidden_size)
            attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        if self.with_last:
            pred_v = torch.cat([v, attn_h, h.view(-1), length]).view(1, -1)
        else:
            pred_v = torch.cat([v, attn_h]).view(1, -1)
        score = self.score(pred_v)

        if self.score_mode == 'concat':
            x = v
        else:
            x = torch.cat([v * (s >= 0.5).type_as(v).expand_as(v),
                           v * (s < 0.5).type_as(v).expand_as(v)])
        x = torch.cat([x, s])

        _, h = self.rnn(x.view(1, 1, -1), h)
        return score, h

    def default_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.seq_hidden_size))


class AttnSeqTimeDecayModel(nn.Module):
    """
    同AttnSeqModel，但增加了依照考试时间远近调整alpha
    """

    def __init__(self, topic_size, seq_hidden_size, k,
                 score_mode, num_layers=1):
        super(AttnSeqTimeDecayModel, self).__init__()
        self.topic_size = topic_size
        self.seq_hidden_size = seq_hidden_size
        self.num_layers = num_layers
        self.score_mode = score_mode
        if self.score_mode == 'concat':
            self.rnn = nn.GRU(topic_size + 1, seq_hidden_size, num_layers)
        else:
            self.rnn = nn.GRU(topic_size * 2 + 1, seq_hidden_size, num_layers)
        self.score = nn.Linear(topic_size + seq_hidden_size, 1)
        self.k = k
        self.initial_h = Variable(torch.zeros(self.num_layers *
                                              self.seq_hidden_size),
                                  requires_grad=True)

    def forward(self, v, s, t, hidden):
        if hidden is None:
            h = self.default_hidden()
            attn_h = self.initial_h
        else:
            vs, hs, ts = hidden
            h = hs[-1:]
            ts = t.expand_as(ts) - ts
            # calculate alpha using dot product
            alpha = torch.mm(vs, v.view(-1, 1)).view(-1)
            alpha, idx = alpha.topk(min(len(alpha), self.k), sorted=False)
            alpha = alpha * ((1 - 1e-7) ** torch.index_select(ts, 0, idx))
            alpha = nn.functional.softmax(alpha.view(1, -1), dim=-1)

            # flatten each h
            hs = hs.view(-1, self.num_layers * self.seq_hidden_size)
            attn_h = torch.mm(alpha, torch.index_select(hs, 0, idx)).view(-1)

        pred_v = torch.cat([v, attn_h]).view(1, -1)
        score = self.score(pred_v)

        if self.score_mode == 'concat':
            x = v
        else:
            x = torch.cat([v * (s >= 0.5).type_as(v).expand_as(v),
                           v * (s < 0.5).type_as(v).expand_as(v)])
        x = torch.cat([x, s])

        _, h = self.rnn(x.view(1, 1, -1), h)
        return score, h

    def default_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.seq_hidden_size))
