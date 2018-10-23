#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
np.random.seed(1234)
from minnimize import Minimize


class TestMinimize():

    def absm(self, a, b):
        return np.abs(a - b).mean()

    def test_min_t_b(self, n_items):
        # 生成数据
        n_stu = 100
        real_b1 = np.random.normal(0, 1, (1, n_items))
        real_t1 = np.random.normal(0, 1, (n_stu, 1))
        real_b2 = np.random.normal(0, 1, (1, n_items))
        real_t2 = np.random.normal(0, 1, (n_stu, 1))
        uv = np.random.random((n_stu, n_items)) < 1.0 / (1.0 + np.exp(-(real_t1 - real_b1))) / (1.0 + np.exp(-(real_t2 - real_b2)))
        # 求能力及难度
        min_t1, min_t2, min_b1, min_b2 = Minimize().fit_t_b(uv, 0.001)
        # 计算误差
        print (self.absm(real_b1, min_b1) + self.absm(real_b2, min_b2) + self.absm(real_t1, min_t1) + self.absm(real_t2, min_t2)) / 4
        return (self.absm(real_b1, min_b1) + self.absm(real_b2, min_b2) + self.absm(real_t1, min_t1) + self.absm(real_t2, min_t2)) / 4

    def plot_min_t_b(self):
        # 根据题目数及误差绘图
        n_items_s = [ii*10 for ii in range(1, 1001)]
        absm_s = []
        for n_items in n_items_s:
            absm_s.append(self.test_min_t_b(n_items))
        plt.plot(n_items_s, absm_s)
        plt.show()

    def test_min_t(self, n_items):
        # 生成数据
        n_stu = 100
        real_b1 = np.random.normal(0, 1, (1, n_items))
        real_t1 = np.random.normal(0, 1, (n_stu, 1))
        real_b2 = np.random.normal(0, 1, (1, n_items))
        real_t2 = np.random.normal(0, 1, (n_stu, 1))
        uv = np.random.random((n_stu, n_items)) < 1.0 / (1.0 + np.exp(-(real_t1 - real_b1))) / (1.0 + np.exp(-(real_t2 - real_b2)))
        # 求能力
        min_t1, min_t2 = Minimize().fit_t(uv, 0.01, real_b1, real_b2)
        # 计算误差
        print (self.absm(real_t1, min_t1) + self.absm(real_t2, min_t2)) / 2
        return (self.absm(real_t1, min_t1) + self.absm(real_t2, min_t2)) / 2

    def plot_min_t(self):
        # 根据题目数及误差绘图
        n_items_s = [ii*10 for ii in range(1, 1001)]
        absm_s = []
        for n_items in n_items_s:
            absm_s.append(self.test_min_t(n_items))
        plt.plot(n_items_s, absm_s)
        plt.show()

if __name__ == "__main__":
    # 能力及难度的误差图
    TestMinimize().plot_min_t_b()
    # 已知难度求能力的误差图
    # TestMinimize().plot_min_t()