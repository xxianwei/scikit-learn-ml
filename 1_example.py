#!/bin/env python
#encoding=utf8

import os
import sys
import math
import re
import common
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

##勤学苦练

##指定字体(可以从windos上拷贝过来使用。当前使用微软雅黑)
font = FontProperties(fname=r"./fonts/msyh.ttf", size=10)
##学习了画图基础、score含义、方差、均值、协方差等计算。以及模型系数、截距（bais）等。

def xyplt(title='披萨价格与直径数据', xlabel='直径(英寸)', ylabel='价格(美元)', code_type='utf8'):
    title = title.decode(code_type)
    xlabel = xlabel.decode(code_type)
    ylabel = ylabel.decode(code_type)
    plt.figure()
    plt.title(title, fontproperties=font)
    plt.xlabel(xlabel, fontproperties=font)
    plt.ylabel(ylabel, fontproperties=font)
    plt.axis([0,25,0,25])
    plt.grid(True)
    return plt

##res = fig.savefig(*args, **kwargs)，
##默认参数都靠后
##第二个参数为字典格式的参数，第一个参数为tuple格式的参数
def save_img(filename='./images/plot2.png', tformat='png'):
    plt.savefig(filename, format=tformat)

if __name__ == "__main__":
    plt = xyplt()
    x = [[6],[8],[10],[14],[18]]
    y = [[7],[9],[13],[17.5],[18]]
    model = LinearRegression()
    model.fit(x, y)
    """
    ##线性模型的相关系数(wi)以及截距(w0)
    print >> sys.stdout, "coef:%s" % model.coef_
    print >> sys.stdout, "coef:%s" % model.intercept_
    """
    x2 = [[2],[5],[7],[11],[16],[22]]
    y2 = model.predict(x2)
    ##作图k为颜色，o为标记，-为线型
    plt.plot(x, y, 'ko')
    plt.plot(x2, y2, 'k-')
    plt.show()
    #save_img()
    ##预测x对应的值
    pred = model.predict(x)
    ##plot将点真实值(k,y[idx])和预测值(k,loss[idx])的距离标志出来
    for idx, k in enumerate(x):
        ##两点连线，将真实的和预测的给出结果
        plt.plot([k,k], [y[idx], pred[idx]], 'b-')
    print >> sys.stdout, "MEAN:%.2f" % (np.mean([6,8,10,14,18]))
    print >> sys.stdout, "MSE:%.2f" % (np.mean((model.predict(x)-y)**2))
    print >> sys.stdout, "VAR:%.2f" % (np.var([6,8,10,14,18], ddof=1))
    print >> sys.stdout, "COV:%.2f" % (np.cov([6,8,10,14,18], [11, 8.5, 15, 18, 11]))[0][0]
    print >> sys.stdout, "COV:%.2f" % (np.cov([6,8,10,14,18], [11, 8.5, 15, 18, 11]))[0][1]
    print >> sys.stdout, "COV:%.2f" % (np.cov([6,8,10,14,18], [11, 8.5, 15, 18, 11]))[1][0]
    print >> sys.stdout, "COV:%.2f" % (np.cov([6,8,10,14,18], [11, 8.5, 15, 18, 11]))[1][1]
    ##cov的参数是x,y的list类型。x本身是list[list]类型
    print >> sys.stdout, "COV1:%f" % np.cov([6,8,10,14,18], [7,9,13,17.5,18])[0][1]
    print >> sys.stdout, "COV0:%f" % np.cov([6,8,10,14,18], [7,9,13,17.5,18])[0][0]
    print >> sys.stdout, "VAR:%f" % np.var(x)
    ##R_score:coefficient of determination,确定系数.表示模型对现实数据的拟合程度
    x_test = [[8],[9],[11],[16],[12]]
    y_test = [[11], [8.5], [15], [18], [11]]
    y_pred = model.predict(x_test)
    y1_tmp = common.array_list_to_list(y_test)
    y2_tmp = common.array_list_to_list(y_pred)
    print >> sys.stdout, "r2_score:%f" % (r2_score(y1_tmp, y2_tmp))

    """
    for i, y in enumerate(y_pred):
        print >> sys.stdout, "%.3f\t%.3f" % (x_test[i][0], y[0])
    """

    sum1 = 0.0
    sum2 = 0.0
    m = np.mean(y1_tmp)
    for i, y1 in enumerate(y1_tmp):
        y2 = y2_tmp[i]
        sum1 += (y2-y1)**2
        sum2 += (y1-m)**2
    print >> sys.stdout, "my_r2_score:%f" % (1-(sum1/sum2))

    print >> sys.stdout, model.score(x_test, y_test)
    save_img("./images/tmp.png")


