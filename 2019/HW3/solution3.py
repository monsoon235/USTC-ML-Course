import numpy as np
import pandas as pd
import scipy.io as scio
import time
import gc


def standardize_img(img: np.ndarray) -> np.ndarray:
    min = img.min(axis=0).reshape((1, img.shape[1]))
    max = img.max(axis=0).reshape((1, img.shape[1]))
    img = (img - min) / (max - min)
    np.nan_to_num(img, copy=False, nan=0)
    return np.hstack([np.ones(shape=(img.shape[0], 1)), img])


class Logistic_Regression_Classifier:
    w: np.ndarray

    def nabla(self, X: np.ndarray, y: np.ndarray, sample_n: int) -> np.ndarray:
        # y=0, coe = -exp/(1+exp)·X
        # y=1, coe = 1/(1+exp)·X
        exp = np.exp(X @ self.w)
        nabla_by_X = -1 / sample_n * (y - exp / (1 + exp)) * X
        nabla = nabla_by_X.sum(axis=0).reshape(self.w.shape)
        return nabla

    def L(self, X: np.ndarray, y: np.ndarray, sample_n: int) -> float:
        exp = np.exp(X @ self.w)
        Li = y * np.log(exp / (1 + exp)) + (1 - y) * np.log(1 / (1 + exp))
        return -1 / sample_n * Li.sum()

    def train(self, X: np.ndarray, y: np.ndarray, step: float = 0.1, n: int = 1000, verbosity: int = 0):
        start = time.time()
        sample_n, d = X.shape
        if verbosity > 0:
            print('=== start training...')
            print('\ttraining binary logistic regression classifier by %d samples' % sample_n)

        self.w = np.full(shape=(d, 1), fill_value=0, dtype=np.float64)
        for i in range(n):
            if verbosity > 0 and (i + 1) % verbosity == 0:
                print('\titeration number: %d, time spent: %d s'
                      % (i + 1, time.time() - start))
                print('\t\tL(w) = %f' % self.L(X=X, y=y, sample_n=sample_n))
            nabla = self.nabla(X=X, y=y, sample_n=sample_n)
            self.w -= step * nabla
            if np.isnan(self.w).sum() != 0 or np.isinf(self.w).sum() != 0:
                # 检测因为不收敛导致的运算溢出
                raise ValueError('divergence detected! iteration number: %d' % i)
        end = time.time()
        if verbosity > 0:
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
    classifier.train(X=train_img, y=train_label, step=1, n=3000, verbosity=100)
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
