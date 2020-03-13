import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# calculate w0 and w1
def calc_linear_model_params(x: pd.Series, y: pd.Series):
    print('[linear regression]')
    # calculate the matrix of linear equations
    sx = x.sum()
    sy = y.sum()
    sx2 = (x ** 2).sum()
    sxy = (x * y).sum()
    A = np.array([
        [len(x), sx],
        [sx, sx2]
    ])
    b = np.array([sy, sxy])
    print('solve following linear equations to get w0 and w1...')
    print('\tn\t* w0 + S(x)\t* w1 = S(Y)')
    print('\tS(x)\t* w0 + S(x^2)\t* w1 = S(xy)')
    print('substitute values...')
    print('\t%f\t* w0 + %f\t* w1 = %f' % (A[0][0], A[0][1], b[0]))
    print('\t%f\t* w0 + %f\t* w1 = %f' % (A[1][0], A[1][1], b[1]))
    # slove equations
    w = np.linalg.solve(A, b)
    print('solution: w0 = %f, w1 = %f\n' % (w[0], w[1]))
    return w[0], w[1]


# calculate w0, w1 and w2
def calc_square_model_params(x: pd.Series, y: pd.Series):
    print('[quadratic regression]')
    # calculate the matrix of linear equations
    sx = x.sum()
    sx2 = (x ** 2).sum()
    sx3 = (x ** 3).sum()
    sx4 = (x ** 4).sum()
    sy = y.sum()
    sxy = (x * y).sum()
    sx2y = ((x ** 2) * y).sum()
    A = np.array([
        [len(x), sx, sx2],
        [sx, sx2, sx3],
        [sx2, sx3, sx4]
    ])
    b = np.array([sy, sxy, sx2y])
    print('solve following linear equations to get w0, w1 and w2...')
    print('\tn\t* w0 + S(x)\t* w1 + S(x^2)\t* w2 = S(y)')
    print('\tS(x)\t* w0 + S(x^2)\t* w1 + S(x^3)\t* w2 = S(xy)')
    print('\tS(x^2)\t* w0 + S(x^3)\t* w1 + S(x^4)\t* w2 = S(x^2*y)')
    print('substitute values...')
    print('\t%f\t* w0 + %f\t* w1 + %f\t* w2 = %f' %
          (A[0][0], A[0][1], A[0][2], b[0]))
    print('\t%f\t* w0 + %f\t* w1 + %f\t* w2 = %f' %
          (A[1][0], A[1][1], A[1][2], b[1]))
    print('\t%f\t* w0 + %f\t* w1 + %f\t* w2 = %f' %
          (A[2][0], A[2][1], A[2][2], b[2]))
    # solve equations
    w = np.linalg.solve(A, b)
    print('solution: w0 = %f, w1 = %f, w2 = %f\n' % (w[0], w[1], w[2]))
    return w[0], w[1], w[2]


if __name__ == "__main__":
    # read data
    xy = pd.read_table('./data', header=None, names=('x', 'y'))
    x = xy['x']
    y = xy['y']

    # calculate linear regression
    w10, w11 = calc_linear_model_params(x, y)


    def f1(x):
        return w10 + w11 * x


    yhat1 = x.map(f1)

    # calculate quadratic regression
    w20, w21, w22 = calc_square_model_params(x, y)


    def f2(x):
        return w20 + w21 * x + w22 * x * x


    yhat2 = x.map(f2)

    # plot
    plt.scatter(x, y, color='r', label='origin data')
    x_range = np.arange(x.min() - 0.1, x.max() + 0.1, step=0.01)
    plt.plot(x_range, [f1(x) for x in x_range], color='g',
             label='linear: y = %+5.4f %+5.4f*x' % (w10, w11))
    plt.plot(x_range, [f2(x) for x in x_range], color='b',
             label='quadratic: y = %+5.4f %+5.4f*x %+5.4f*x^2' % (w20, w21, w22))
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./plot1.2.eps', dpi=400, size=(800, 600))
    plt.savefig('./plot1.2.png', dpi=400, size=(800, 600))

    # choose better model
    loss1 = ((yhat1 - y) ** 2).sum()
    loss2 = ((yhat2 - y) ** 2).sum()
    print('total loss of linear regression:', loss1)
    print('total loss of quadratic regression:', loss2)
    print()
    loss = loss1 - loss2
    if loss < -1e-10:
        print('linear regression better\nw0=%f, w1=%f' % (w10, w11))
    elif loss > 1e-10:
        print('quadratic regression better\nw0=%f, w1=%f, w2=%f' %
              (w20, w21, w22))
    else:
        print('same good')
        print('linear regression: w0=%f, w1=%f' % (w10, w11))
        print('quadratic regression: w0=%f, w1=%f, w2=%f' % (w20, w21, w22))
