# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

from datetime import datetime
from collections import namedtuple, defaultdict
from operator import itemgetter
from io import open
import os
import time
import json
import numpy as np
from yata.fields import Words, Categorical
from yata.loaders import TableLoader
from random import sample, randint, shuffle, seed

# 做题记录
Record = namedtuple('Record', ['user', 'school', 'topic', 'exam',
                               'score', 'time'])

# 学生做题序列，其中bias用于执行考试内打乱
Item = namedtuple('Item', ['topic', 'score', 'time', 'bias'])


class Dataset:
    def __init__(self, random_level=1):
        """
        构造一个空的Dataset
        :param random_level: 获取序列时随机打乱的程度，0为不打乱，1为考试内打乱，2为全
        部打乱。默认1
        """
        self.topics = set()
        self.exams = set()
        self.users = set()
        self.schools = defaultdict(set)
        self.records = list()
        self.user_school_map = dict()
        self.topic_exam_map = dict()
        self.random_level = random_level

    @staticmethod
    def from_matrix(filename):
        f = open(filename, encoding='utf-8')
        d = Dataset()
        for line in f:
            desc, seq = line.strip().split('\t')
            scl, exam, grd = desc.strip().split('@')
            # 去除异常考试
            if exam == 'e061cfc1-86e9-4486-abc7-0630d0c1ad2d':
                continue
            if exam == '8c7a9923-0856-4fed-b326-c0f124129b86':
                continue
            data_fields = seq.strip().split(' ')
            exam_time = int(data_fields[0])
            for record in data_fields[5:]:
                user, topic, score, std_score = record.split('@')
                score, std_score = float(score), float(std_score)
                r = Record(user, scl, topic, exam,
                           score / std_score, exam_time)
                d._insert(r)
        return d

    @staticmethod
    def from_records(dirname):
        d = Dataset()

        record_f = open(os.path.join(dirname, 'records.txt'), encoding='utf-8')
        school_f = open(os.path.join(dirname, 'schools.txt'), encoding='utf-8')
        exam_f = open(os.path.join(dirname, 'exams.txt'), encoding='utf-8')

        topic_exam_map = dict()
        exam_info = dict()
        student_scl_map = dict()

        for line in exam_f:
            fields = line.strip().split(' ')
            exam_id, exam_type, exam_time, _ = fields[0].split(',')
            exam_id = exam_id + '_' + exam_type
            exam_time = \
                int(time.mktime(datetime.strptime(exam_time,
                                                  '%Y-%m-%d').timetuple()))
            exam_info[exam_id] = exam_time
            for topic in fields[1:]:
                topic_exam_map[topic] = exam_id

        for line in school_f:
            fields = line.strip().split(' ')
            school_id = fields[0]
            for student in fields[1:]:
                student_scl_map[student] = school_id

        exam_by_school = defaultdict(set)

        for line in record_f:
            fields = line.strip().split(' ')
            student_id, _, _ = fields[0].split(',')
            if student_id not in student_scl_map:
                continue
            for item in fields[1:]:
                topic_id, score = item.split(',')
                score = float(score)
                scl_id = student_scl_map[student_id]
                exam_id = topic_exam_map[topic_id]
                r = Record(student_id, scl_id, topic_id, exam_id,
                           score, exam_info[exam_id])
                exam_by_school[scl_id].add(exam_id)
                d._insert(r)

        return d

    def select(self, filter):
        """
        选择满足条件的记录集合，分别返回满足和不满足的两个dataset
        :param filter: 判断条件（函数）
        :return: selected, others
        """
        selected = Dataset()
        others = Dataset()
        for r in self.records:
            if filter(r):
                selected._insert(r)
            else:
                others._insert(r)
        return selected, others

    def split(self):
        """
        划分数据为训练集、测试集（新学生、新考试、学生考试都未出现）
        :return: train, user, exam, new
        """
        train = Dataset()
        user = Dataset()
        exam = Dataset()
        new = Dataset()

        train_exams = []
        schools = dict()
        for s in self.schools:
            schools[s] = sorted(list(self.schools[s]),
                                key=itemgetter(1))
        for s in schools:
            train_exams.extend([x[0] for x in schools[s][:-1]])
        train_exams = set(train_exams)
        train_users = sample(sorted(self.users), 0.9)
        train_users = set(train_users)

        for r in self.records:
            if r.exam in train_exams and r.user in train_users:
                # train set
                train._insert(r)
            elif r.exam in train_exams:
                # new user
                user._insert(r)
            elif r.user in train_users:
                # new exam
                exam._insert(r)
            else:
                # completely new record
                new._insert(r)

        train.random_level = self.random_level
        user.random_level = self.random_level
        exam.random_level = self.random_level
        new.random_level = self.random_level

        return train, user, exam, new

    def split_future(self, frac, rand_seed=324):
        seq = self.get_seq()
        train_data = Dataset()
        test_data = Dataset()
        seed(rand_seed)

        for user in seq:
            school = self.user_school_map[user]
            u_seq = seq[user]
            train_len = int(frac * len(u_seq))
            for topic, score, time, _ in u_seq[:train_len]:
                exam = self.topic_exam_map[topic]
                train_data._insert(Record(user, school, topic, exam,
                                          score, time))
            for topic, score, time, _ in u_seq[train_len:]:
                exam = self.topic_exam_map[topic]
                test_data._insert(Record(user, school, topic, exam,
                                         score, time))

        return train_data, test_data

    def split_user(self, frac, rand_seed=101):
        seed(rand_seed)
        train_users = sample(sorted(self.users),
                             int(len(self.users) * frac))
        train_users = set(train_users)
        train_data = Dataset()
        test_data = Dataset()
        for r in self.records:
            if r.user in train_users:
                train_data._insert(r)
            else:
                test_data._insert(r)

        return train_data, test_data

    def get_seq(self):
        """
        返回每个学生的做题序列，根据设定的打乱程度（random_level，0为不打乱，1为考试内打
        乱，2为全部打乱）对序列进行随机打乱
        :return: 一个学生到该学生做题记录（Item）序列的字典
        """
        seq = defaultdict(list)
        for r in self.records:
            seq[r.user].append(Item(r.topic, r.score,
                                    r.time, randint(-5000, 5000)))
        for user in seq:
            if self.random_level == 1:
                seq[user].sort(key=lambda x: x.time + x.bias)
            elif self.random_level == 2:
                shuffle(seq[user])
        return seq

    def get_dict(self):
        """
        返回学生、题目的序号以及反查表
        :return: 学生序号、序号反查、题目序号、序号反查
        """
        user_dic = {}
        topic_dic = {}
        user_inv_dic = {}
        topic_inv_dic = {}
        for i, user in enumerate(sorted(self.users)):
            user_dic[user] = i + 1
            user_inv_dic[i + 1] = user
        for i, topic in enumerate(sorted(self.topics)):
            topic_dic[topic] = i + 1
            topic_inv_dic[i + 1] = topic
        return user_dic, user_inv_dic, topic_dic, topic_inv_dic

    def save(self, filename):
        f = open(filename, 'w')
        json.dump(self.records, f)
        f.close()

    def load(self, filename):
        f = open(filename)
        records = json.load(f)
        for r in records:
            self._insert(Record(*r))

    def _insert(self, r):
        self.topics.add(r.topic)
        self.exams.add(r.exam)
        self.users.add(r.user)
        self.schools[r.school].add((r.exam, r.time))
        self.user_school_map[r.user] = r.school
        self.topic_exam_map[r.topic] = r.exam
        self.records.append(r)


def get_dataset(type, random_level=0):
    """
    返回数据集
    :param type: {full,some}[_test]
    :param random_level: 0为不打乱，1为考试内打乱，2为全部打乱，默认1
    :return: 对应数据集
    """
    some_schools = ['2300000001000000032',
                    '2300000001000674122',
                    '4444000020000000449',
                    '2300000001000649665',
                    '2300000001000053674',
                    '2300000001000649702']
    some_test_schools = ['4444000020000000470']

    if type.startswith('full'):
        if type.endswith('test'):
            rv = Dataset.from_records('data/test')
        else:
            rv = Dataset.from_records('data/full')
    elif type.startswith('some'):
        d = Dataset.from_matrix('data/02.10.matrix')
        if type.endswith('test'):
            rv, _ = d.select(lambda r: r.school in some_test_schools)
        else:
            rv, _ = d.select(lambda r: r.school in some_schools)
    else:
        rv = Dataset()
        rv.load(type)
    rv.random_level = random_level
    return rv


def load_embedding(filename):
    f = open(filename, encoding='utf-8')
    wcnt, emb_size = next(f).strip().split(' ')
    wcnt = int(wcnt)
    emb_size = int(emb_size)

    words = []
    embs = []
    for line in f:
        fields = line.strip().split(' ')
        word = fields[0]
        emb = np.array([float(x) for x in fields[1:]])
        words.append(word)
        embs.append(emb)

    embs = np.asarray(embs)
    return wcnt, emb_size, words, embs


def get_topics(type, words):
    if type.startswith('some'):
        feature_file = 'data/features.dump.some'
    else:
        feature_file = 'data/features.dump.full'

    to_categorical = Categorical('</s>')
    to_categorical.load_dict(words)
    topic_fields = {
        '2->content': to_categorical(Words(':', null='</s>')),
    }
    topics = TableLoader(feature_file, with_header=False,
                         key=0, fields=topic_fields, index=['content'])
    return topics


if __name__ == '__main__':
    # data = Dataset()
    # data.load('data/raw50/full_sampled.json')
    data = get_dataset('full', random_level=2)

    # print('#topic', len(data.topics))
    # print('#exam', len(data.exams))
    # print('#users', len(data.users))
    # print('#school', len(data.schools))
    # print('#records', len(data.records))

    data = data.get_seq()
    users = list(data)
    shuffle(users)
    for user in users[:20]:
        seq = data[user]
        for item in seq:
            print(item.topic, item.score, sep=',', end=' ')
        print()

    # f = open('data/id_dict', 'w')
    # json.dump(data.get_dict(), f)
    # f.close()

    '''
    print(data.topics)

    trainset, test_user, test_exam, test_new = data.split()

    print('trainset:')
    print('#topic', len(trainset.topics))
    print('#exam', len(trainset.exams))
    print('#users', len(trainset.users))
    print('#school', len(trainset.schools))
    print('#records', len(trainset.records))
    print('user:')
    print('#topic', len(test_user.topics))
    print('#exam', len(test_user.exams))
    print('#users', len(test_user.users))
    print('#school', len(test_user.schools))
    print('#records', len(test_user.records))
    print('exam:')
    print('#topic', len(test_exam.topics))
    print('#exam', len(test_exam.exams))
    print('#users', len(test_exam.users))
    print('#school', len(test_exam.schools))
    print('#records', len(test_exam.records))
    print('new:')
    print('#topic', len(test_new.topics))
    print('#exam', len(test_new.exams))
    print('#users', len(test_new.users))
    print('#school', len(test_new.schools))
    print('#records', len(test_new.records))
    '''
