import numpy as np
import pandas as pd
import scipy.io as scio
import time
import gc


def standardize_img(img: np.ndarray):
    min = img.min(axis=0).reshape((1, img.shape[1]))
    max = img.max(axis=0).reshape((1, img.shape[1]))
    img = (img - min) / (max - min)
    np.nan_to_num(img, copy=False, nan=0)
    return np.hstack([np.ones(shape=(img.shape[0], 1)), img])


class Logistic_Regression_Classifier:
    w: np.ndarray

    def train(self, X: np.ndarray, y: np.ndarray, step: float = 0.1, n: int = 1000, verbose: bool = False):
        start = time.time()
        sample_n, d = X.shape
        if verbose:
            print('=== start training...')
            print('\ttraining binary logistic regression classifier by %d samples' % sample_n)

        self.w = np.full(shape=(d, 1), fill_value=0, dtype=np.float64)
        for i in range(n):
            if verbose and (i + 1) % 100 == 0:
                print('\titeration number: %d, time spent: %d s'
                      % (i + 1, time.time() - start))
            # 若 y=0, 则 nabla = -exp/(1+exp)·X
            # 若 y=1, 则 nabla = 1/(1+exp)·X
            exp = np.exp(X @ self.w)
            coe = -exp / (1 + exp)  # y=0 时 nabla 的系数
            nabla_by_X = (coe + y) * X  # +y 是简化 y=1 时 nabla 计算的手段
            nabla = - nabla_by_X.sum(axis=0).reshape(self.w.shape)
            self.w -= step * nabla
            if np.isnan(self.w).sum() != 0 or np.isinf(self.w).sum() != 0:
                # 检测因为不收敛导致的运算溢出
                raise ValueError('divergence detected! iteration number: %d' % i)
        end = time.time()
        print('=== training finished in %d s' % (end - start))
        gc.collect()

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = X @ self.w
        # X@w>0 即判为属于 1 类
        return (prob > 0).astype(int)


if __name__ == '__main__':
    train_img: np.ndarray = scio.loadmat('hw3_lr/train_imgs.mat')['train_img'].toarray()
    train_label: np.ndarray = scio.loadmat('hw3_lr/train_labels.mat')['train_label'].toarray().T
    test_img: np.ndarray = scio.loadmat('hw3_lr/test_imgs.mat')['test_img'].toarray()
    test_label: np.ndarray = scio.loadmat('hw3_lr/test_labels.mat')['test_label'].toarray().T

    # 归一化，并加上 x0=1
    train_img = standardize_img(train_img)
    test_img = standardize_img(test_img)
    # label 1 和 2 分别变为 0 和 1
    train_label = train_label - 1
    test_label = test_label - 1

    classifier = Logistic_Regression_Classifier()
    classifier.train(X=train_img, y=train_label, step=0.0005, n=600, verbose=True)
    pred_label = classifier.predict(X=test_img)

    # 假设 2 为正例，1 为反例
    print("\nNotice: assuming '2' is positive and '1' is negative")

    tp = np.sum((test_label == 1) & (pred_label == 1))
    fn = np.sum((test_label == 1) & (pred_label == 0))
    fp = np.sum((test_label == 0) & (pred_label == 1))
    tn = np.sum((test_label == 0) & (pred_label == 0))
    confusion_matrix = pd.DataFrame(
        data=[
            [tp, fn],
            [fp, tn]
        ],
        index=pd.Index(data=[2, 1], name='Actual'),
        columns=pd.Index(data=[2, 1], name='Predicted'),
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
