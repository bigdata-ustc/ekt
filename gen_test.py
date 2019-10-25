from dataprep import Dataset, get_dataset, Record
from random import sample

full = get_dataset('full')
full.random_level = 0
seq = full.get_seq()

for frac in [60, 70, 80, 90]:
    train = get_dataset('data/raw50/future.train.%d' % frac)
    train_topics = set(train.topics)
    test = Dataset()
    cold_test = Dataset()
    train_seq = train.get_seq()
    for user in train_seq:
        L = int(len(train_seq[user]) / frac * (100 - frac))
        topics = set(x.topic for x in seq[user])
        same = topics & train_topics - set(x.topic for x in train_seq[user])
        diff = topics - train_topics
        pop = same | diff
        print(len(same), len(diff))
        L = max(5, L - len(diff))
        selected = set(sample(list(same), L)) | diff

        for topic, score, time, _ in seq[user]:
            r = Record(user, full.user_school_map[user],
                       topic, full.topic_exam_map[topic],
                       score, time)
            if topic in diff:
                cold_test._insert(r)
            if topic in selected:
                test._insert(r)

    test.save('data/testsets2/future.test.%d' % frac)
    cold_test.save('data/testsets2/coldstart.%d' % frac)
