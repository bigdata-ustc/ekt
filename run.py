# -*- coding: utf-8 -*-

"""
执行模型训练、测试等
"""

import argparse
import os
import sys
import logging

from model import *
from command import *
from util import *

commands = ['config', 'train', 'test', 'testfuture', 'testseq', 'stat',
            'statoverall', 'predict', 'trend', 'kt', 'corr', 'at', 'corr2']
models = ['RNN', 'Attn', 'RA', 'RADecay', 'LSTMM', 'LSTMA', 'DKT',
          'DKNM', 'DKNA', 'EKTM', 'EKTA', 'DKVMN']


class Config:
    def __init__(self, parser):
        subs = parser.add_subparsers(title='models available', dest='model')
        subs.required = True
        group_options = set()
        for model in models:
            sub = subs.add_parser(model, formatter_class=parser_formatter)
            group = sub.add_argument_group('setup')
            Model = get_class(model)
            Model.add_arguments(group)
            for action in group._group_actions:
                group_options.add(action.dest)

            def save(args):
                for file in os.listdir(args.workspace):
                    if file.endswith('.json'):
                        os.remove(os.path.join(args.workspace, file))
                model = args.model
                Model = get_class(model)
                setup = {name: value for (name, value) in args._get_kwargs()
                         if name in group_options}
                conf = os.path.join(args.workspace,
                                    str(model) + '.json')
                m = Model(setup)
                print('model: %s, setup: %s' % (model, str(m.args)))
                save_config(m, conf)

            sub.set_defaults(func=save)

    def run(self, args):
        pass


class Train:
    def __init__(self, parser):
        parser.add_argument('-N', '--epochs', type=int, default=1,
                            help='number of epochs to train')
        parser.add_argument('-d', '--dataset', required=True)
        parser.add_argument('-s', '--split_method',
                            choices=['future', 'user', 'old', 'none'])
        parser.add_argument('-f', '--frac', default=0.1, type=float,
                            help='train data fraction')
        parser.add_argument('-rl', '--random_level', default=0, type=int,
                            help='random level')
        parser.add_argument('-l', '--loss', default='mse',
                            choices=['mse', 'cross_entropy'])
        parser.add_argument('--print_every', type=int, default=10,
                            help='logging interval')
        parser.add_argument('--save_every', type=int, default=1000,
                            help='saving interval')
        parser.add_argument('-ik', '--input_knowledge', action='store_true')
        parser.add_argument('-it', '--input_text',action='store_true')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        # train(model, args)
        trainn(model, args)


class Test:
    def __init__(self, parser):
        parser.add_argument('-e', '--snapshot',
                            help='model snapshot to test with')
        parser.add_argument('-d', '--dataset',
                            required=True)
        parser.add_argument('-t', '--test_as_seq', action='store_true',
                            help='test sequences using output scores')
        parser.add_argument('-o', '--test_on_one', action='store_true',
                            help='test on next one')
        parser.add_argument('-z', '--test_on_last', action='store_true',
                            help='test last')
        parser.add_argument('-r', '--ref_len', type=int, default=0,
                            help='length of sequence with true scores')
        parser.add_argument('-rs', '--ref_set')
        parser.add_argument('-s', '--split_method',
                            choices=['future', 'user', 'old', 'none'])
        parser.add_argument('-f', '--frac', default=0.1, type=float,
                            help='train data fraction')
        parser.add_argument('-rl', '--random_level', default=0, type=int,
                            help='random level')
        parser.add_argument('-l', '--loss', default='mse',
                            choices=['mse', 'cross_entropy'])
        parser.add_argument('--print_every', type=int, default=10,
                            help='logging interval')
        parser.add_argument('-ik', '--input_knowledge', action='store_true')
        parser.add_argument('-it', '--input_text',action='store_true')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        test(model, args)


class Testseq:
    def __init__(self, parser):
        parser.add_argument('-e', '--snapshot',
                            help='model snapshot to test with')
        parser.add_argument('-d', '--dataset',
                            required=True)
        parser.add_argument('-r', '--ref_len', type=int, default=0,
                            help='length of sequence with true scores')
        parser.add_argument('-rs', '--ref_set')
        parser.add_argument('-s', '--split_method',
                            choices=['future', 'user', 'old', 'none'])
        parser.add_argument('-f', '--frac', default=0.1, type=float,
                            help='train data fraction')
        parser.add_argument('-rl', '--random_level', default=0, type=int,
                            help='random level')
        parser.add_argument('-l', '--loss', default='mse',
                            choices=['mse', 'cross_entropy'])
        parser.add_argument('--print_every', type=int, default=10,
                            help='logging interval')
        parser.add_argument('-ik', '--input_knowledge', action='store_true')
        parser.add_argument('-it', '--input_text',action='store_true')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        # test(model, args)
        testseq(model, args)


class Testfuture:
    def __init__(self, parser):
        parser.add_argument('-e', '--snapshot',
                            help='model snapshot to test with')
        parser.add_argument('-d', '--dataset',
                            required=True)
        parser.add_argument('-r', '--ref_len', type=int, default=0,
                            help='length of sequence with true scores')
        parser.add_argument('-rs', '--ref_set')
        parser.add_argument('-s', '--split_method',
                            choices=['future', 'user', 'old', 'none'])
        parser.add_argument('-f', '--frac', default=0.1, type=float,
                            help='train data fraction')
        parser.add_argument('-rl', '--random_level', default=0, type=int,
                            help='random level')
        parser.add_argument('-l', '--loss', default='mse',
                            choices=['mse', 'cross_entropy'])
        parser.add_argument('--print_every', type=int, default=10,
                            help='logging interval')
        parser.add_argument('-ik', '--input_knowledge', action='store_true')
        parser.add_argument('-it', '--input_text',action='store_true')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        # test(model, args)
        testfuture(model, args)


class Stat:
    def __init__(self, parser):
        parser.add_argument('result_file')
        parser.add_argument('-a', '--with_auc', action='store_true')
        parser.add_argument('-r', '--round_score', action='store_true')
        parser.add_argument('-s', '--short', action='store_true')

    def run(self, args):
        stat(open(args.result_file), args.with_auc, args.round_score)


class Statoverall:
    def __init__(self, parser):
        parser.add_argument('result_file')
        parser.add_argument('-a', '--with_auc', action='store_true')
        parser.add_argument('-r', '--round_score', action='store_true')
        parser.add_argument('-s', '--short', action='store_true')

    def run(self, args):
        stat_overall(open(args.result_file), args.with_auc, args.round_score)


class Trend:
    def __init__(self, parser):
        parser.add_argument('-e', '--snapshot',
                            help='model snapshot to test with')
        parser.add_argument('-d', '--dataset', required=True)

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        if use_cuda:
            model.cuda()
        trend(model, args.dataset)


class Predict:
    def __init__(self, parser):
        parser.add_argument('-e', '--snapshot',
                            help='model snapshot to test with')

    def run(self, args):
        for name in os.listdir(args.workspace):
            if name.endswith('.json'):
                Model = get_class(name.split('.')[0])
                config = os.path.join(args.workspace, name)
                break
        else:
            print('you must run config first!')
            sys.exit(1)

        model = load_config(Model, config)
        predict(model, args)


def get_class(name):
    return globals()[name[0].upper() + name[1:]]


if __name__ == '__main__':
    for command in commands:
        sub = subparsers.add_parser(command, formatter_class=parser_formatter)
        subcommand = get_class(command)(sub)
        sub.set_defaults(func=subcommand.run)

    args = parser.parse_args()
    workspace = args.workspace
    try:
        os.makedirs(os.path.join(workspace, 'snapshots'))
        os.makedirs(os.path.join(workspace, 'results'))
        os.makedirs(os.path.join(workspace, 'logs'))
    except OSError:
        pass

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logFormatter = ColoredFormatter('%(levelname)s %(asctime)s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    fileFormatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

    if args.command != 'config':
        fileHandler = logging.FileHandler(os.path.join(workspace, 'logs',
                                                       args.command + '.log'))
        fileHandler.setFormatter(fileFormatter)
        logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    try:
        args.func(args)
    except KeyboardInterrupt:
        logging.warn('cancelled by user')
    except Exception as e:
        import traceback
        sys.stderr.write(traceback.format_exc())
        logging.warn('exception occurred: %s', e)
