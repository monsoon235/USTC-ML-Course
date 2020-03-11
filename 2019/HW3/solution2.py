import numpy as np
import pandas as pd
import os
import re
import gc
import time
import math
from typing import Tuple

splitter = re.compile(r'\s+')
spam_pattern = re.compile(r'^spmsg.*\.txt$')


# 预处理邮件，
def preprocess_mail(path: str) -> Tuple[str]:
    with open(path, 'r') as mail:
        content = mail.read()
        tokens = splitter.split(content)
        tokens = tuple(t for t in tokens if t.isalpha())
    return tokens


# 通过文件名判断是否是 spam
# spam is 1
# non-spam is 0
def is_spam(file_name: str) -> int:
    if spam_pattern.match(file_name):
        return 1
    else:
        return 0


def read_data(path: str) -> pd.DataFrame:
    files = os.listdir(path)
    data = pd.DataFrame(
        data={
            'tokens': tuple(preprocess_mail(path + f) for f in files),
            'is_spam': tuple(is_spam(f) for f in files),
        }
    )
    data.index.name = 'mail no.'
    return data


class Naive_Bayes_Classifier_For_Mail:
    token_set: set  # 训练集中出现的所有 token
    class_set: set  # 所有类别
    p_by_class: dict  # 每种类别出现概率
    p_by_token_in_class: dict  # 在某个类别中，某个 token 出现的概率

    def __init__(self):
        self.token_set = set()
        self.class_set = set()
        self.p_by_class = {}
        self.p_by_token_in_class = {}

    def train(self, X: pd.Series, y: pd.Series, verbose: bool = False):
        start = time.time()
        if verbose:
            print('=== start training...')

        assert len(X) == len(y)
        n = len(X)  # 训练集的大小

        for tokens in X:
            self.token_set.update(tokens)
        token_n = len(self.token_set)  # 训练集中出现的 token 总数
        self.class_set = set(y)  # 所有类别

        # 按类别划分样本
        grouped_by_class = X.groupby(by=y)
        samples_by_class = grouped_by_class.agg(tuple)  # 每种类别的样本

        if verbose:
            print('\ttraining naive bayes classifier for spam mail at %d samples' % n)
            print('\tspam mail count: %d' % len(samples_by_class[1]))
            print('\tnon-spam mail count: %d' % len(samples_by_class[0]))
            print('\ttoken count in all samples: %d' % len(self.token_set))

        self.p_by_token_in_class = {t: {} for t in self.token_set}

        for c in self.class_set:
            samples = samples_by_class[c]  # 此类的所有样本
            self.p_by_class[c] = len(samples) / n  # 此类样本在所有样本中的出现概率
            nc = np.sum([len(s) for s in samples])  # 此类样本 token 总数
            for t in self.token_set:
                # token 在此类样本中的出现次数
                nck = np.sum([s.count(t) for s in samples])
                self.p_by_token_in_class[t][c] = (nck + 1) / (nc + token_n)

        end = time.time()
        if verbose:
            print('=== training finished in %d s' % (end - start))
        gc.collect()

    def predict(self, X: pd.Series) -> pd.Series:
        result = pd.Series(
            data=np.empty(shape=X.shape, dtype=int),
            index=X.index,
        )
        for i, mail in enumerate(X):
            max_prob = - math.inf
            for c in self.class_set:
                # y_hat = self.p_by_class[c] * np.prod([
                #     # 所有出现在 mail 和 训练集中的 token
                #     self.p_by_token_in_class[t][c]
                #     for t in mail if t in self.token_set
                # ])
                # 取 log 把连乘变加法之后效果变好，可能是精度问题
                # p 大致都在 1e-5 数量级
                y_hat = np.log(self.p_by_class[c]) + np.log([
                    # 所有出现在 mail 和 训练集中的 token
                    self.p_by_token_in_class[t][c]
                    for t in mail if t in self.token_set
                ]).sum()
                if y_hat > max_prob:
                    max_prob = y_hat
                    result[i] = c
        gc.collect()
        return result


if __name__ == '__main__':
    train_mails = read_data('hw3-nb/train-mails/')
    test_mails = read_data('hw3-nb/test-mails/')
    classifier = Naive_Bayes_Classifier_For_Mail()
    classifier.train(
        X=train_mails.loc[:, 'tokens'],
        y=train_mails.loc[:, 'is_spam'],
        verbose=True,
    )
    predicted = classifier.predict(
        X=test_mails.loc[:, 'tokens']
    )
    actual = test_mails.loc[:, 'is_spam']

    # spam 为正例，non-spam 为反例
    tp = np.sum((actual == 1) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    tn = np.sum((actual == 0) & (predicted == 0))
    confusion_matrix = pd.DataFrame(
        data=[
            [tp, fn],
            [fp, tn],
        ],
        index=pd.Index(data=['Spam', 'Non-spam'], name='Actual'),
        columns=pd.Index(data=['Spam', 'Non-spam'], name='Predicted')
    )
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 / (1 / precision + 1 / recall)
    print('\nConfusion matrix =')
    print(confusion_matrix)
    print()
    print('Precision =', precision)
    print('Recall =', recall)
    print('F1 score =', f1)
