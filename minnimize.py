#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy.optimize import minimize


class Minimize():

    # 由结果y求能力t，难度b
    # 输入：结果矩阵y，二维矩阵
    # 输出：能力t1,能力t2，难度b1，难度b2，均为二维矩阵
    def fit_t_b(self, y, w):
        start1 = y.shape[0]
        start2 = y.shape[0]*2
        start3 = y.shape[0]*2+y.shape[1]
        # 变量赋初始值，未知数由能力1，能力2，难度1，难度2拼接而成。
        x0_init = np.random.normal(0, 1, 2*(sum(y.shape)))
        # 变量取值范围
        bnds = ((-3, 3), (-3, 3))*(sum(y.shape))

        # 负似然函数及梯度
        def f(x):
            # 从变量取出能力1，组成二维矩阵形式
            t1 = np.array(x[:start1]).reshape(-1, 1)
            # 从变量取出难度1，组成二维矩阵形式
            b1 = np.array([[x[start2:start3]]])
            # 从变量取出能力2，组成二维矩阵形式
            t2 = np.array(x[start1:start2]).reshape(-1, 1)
            # 从变量取出难度2，组成二维矩阵形式
            b2 = np.array([[x[start3:]]])
            # 计算概率p1，p2
            p1 = (1. / (1 + np.exp(-(t1 - b1)))).reshape(y.shape)
            p2 = (1. / (1 + np.exp(-(t2 - b2)))).reshape(y.shape)
            # 计算负似然函数
            cost = -(y * (np.log(p1) + np.log(p2)) + (1 - y) * np.log(1 - p1 * p2)).sum() / y.shape[1] + w * ((np.array([x]) ** 2).sum())
            # 计算能力1的梯度，梯度就是一阶导数
            gradient_t1 = -((y - p1 * p2) * (1 - p1) / (1 - p1 * p2)).sum(axis=1) / y.shape[1] + 2 * w * np.array(x[:start1])
            # 计算能力2的梯度
            gradient_t2 = -((y - p1 * p2) * (1 - p2) / (1 - p1 * p2)).sum(axis=1) / y.shape[1] + 2 * w * np.array(
                x[start1:start2])
            # 计算难度1的梯度
            gradient_b1 = ((y - p1 * p2) * (1 - p1) / (1 - p1 * p2)).sum(axis=0) / y.shape[1] + 2 * w * np.array(
                x[start2:start3])
            # 计算难度2的梯度
            gradient_b2 = ((y - p1 * p2) * (1 - p2) / (1 - p1 * p2)).sum(axis=0) / y.shape[1] + 2 * w * np.array(x[start3:])
            # 梯度
            gradient = np.concatenate((gradient_t1, gradient_t2, gradient_b1, gradient_b2))
            # 返回负似然函数及梯度
            return cost, gradient
        # 第一个参数：函数，第二个参数：变量初值，第三个参数：变量范围，第四个参数：算法，第五个参数：使用梯度
        res = minimize(f, x0_init, bounds=bnds, method='L-BFGS-B', jac=True)
        # 将变量结果拆分为难度t1，t2，b1，b2，均为二维数组
        min_t1 = np.array(res.x[:start1]).reshape(y.shape[0], 1)
        min_b1 = np.array([[res.x[start2:start3]]])
        min_t2 = np.array(res.x[start1:start2]).reshape(y.shape[0], 1)
        min_b2 = np.array([[res.x[start3:]]])
        # 返回难度t1，t2，b1，b2，均为二维数组
        return min_t1, min_t2, min_b1, min_b2

    # 由结果y，难度b求能力t
    # 输入：结果矩阵y，难度b1，难度b2，均为二维矩阵
    # 输出：能力t1,能力t2，均为二维矩阵
    def fit_t(self, y, w, b1, b2):
        start1 = y.shape[0]
        # 变量赋初始值，未知数由能力1，能力2拼接而成。
        x0_init = np.random.normal(0, 1, 2 * y.shape[0])
        # 变量取值范围（-3,3）
        bnds = ((-3, 3), (-3, 3)) * (y.shape[0])

        # 负似然函数及梯度
        def f(x):
            # 从变量取出能力1，组成二维矩阵形式
            t1 = np.array(x[:start1]).reshape(-1, 1)
            # 从变量取出能力2，组成二维矩阵形式
            t2 = np.array(x[start1:]).reshape(-1, 1)
            # 计算概率p1，p2
            p1 = (1. / (1 + np.exp(-(t1 - b1)))).reshape(y.shape)
            p2 = (1. / (1 + np.exp(-(t2 - b2)))).reshape(y.shape)
            # 计算负似然函数
            cost = -(y * (np.log(p1) + np.log(p2)) + (1 - y) * np.log(1 - p1 * p2)).sum() / y.shape[1] + w * ((np.array([x]) ** 2).sum())
            # 计算能力1的梯度，梯度就是一阶导数
            gradient_t1 = -((y - p1 * p2) * (1 - p1) / (1 - p1 * p2)).sum(axis=1) / y.shape[1] + 2 * w * np.array(x[:start1])
            # 计算能力2的梯度
            gradient_t2 = -((y - p1 * p2) * (1 - p2) / (1 - p1 * p2)).sum(axis=1) / y.shape[1] + 2 * w * np.array(x[start1:])
            # 梯度
            gradient = np.concatenate((gradient_t1, gradient_t2))
            # 返回负似然函数及梯度
            return cost, gradient

        # 第一个参数：函数，第二个参数：变量初值，第三个参数：变量范围，第四个参数：算法，第五个参数：使用梯度
        res = minimize(f, x0_init, bounds=bnds, method='L-BFGS-B', jac=True)
        # 将变量结果拆分为难度t1，t2，均为二维数组
        min_t1 = np.array(res.x[:start1]).reshape(y.shape[0], 1)
        min_t2 = np.array(res.x[start1:]).reshape(y.shape[0], 1)
        # 返回难度t1，t2，b1，b2，均为二维数组
        return min_t1, min_t2
