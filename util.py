import json
import torch
import os
import logging


def save_config(obj, path):
    f = open(path, 'w')
    json.dump(obj.args, f, indent='  ')
    f.write('\n')
    f.close()


def load_config(Model, path):
    f = open(path, 'r')
    return Model(json.load(f))


def save_snapshot(model, ws, id):
    filename = os.path.join(ws, 'snapshots', 'model.%s' % str(id))
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def load_snapshot(model, ws, id):
    filename = os.path.join(ws, 'snapshots', 'model.%s' % str(id))
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def load_last_snapshot(model, ws):
    last = 0
    for file in os.listdir(os.path.join(ws, 'snapshots')):
        if 'model.' in file:
            epoch = int(file.split('.')[1])
            if epoch > last:
                last = epoch
    if last > 0:
        load_snapshot(model, ws, last)
    return last


def open_result(ws, name, id):
    return open(os.path.join(ws, 'results', '%s.%s' %
                             (name, str(id))), 'w')


use_cuda = torch.cuda.is_available()


def Variable(*args, **kwargs):
    v = torch.autograd.Variable(*args, **kwargs)
    if use_cuda:
        v = v.cuda()
    return v

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored(text, color, bold=False):
    if bold:
        return bcolors.BOLD + color + text + bcolors.ENDC
    else:
        return color + text + bcolors.ENDC


LOG_COLORS = {
    'WARNING': bcolors.WARNING,
    'INFO': bcolors.OKGREEN,
    'DEBUG': bcolors.OKBLUE,
    'CRITICAL': bcolors.WARNING,
    'ERROR': bcolors.FAIL
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, datefmt, use_color=True):
        logging.Formatter.__init__(self, msg, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in LOG_COLORS:
            record.levelname = colored(record.levelname[0],
                                       LOG_COLORS[record.levelname])
        return logging.Formatter.format(self, record)
