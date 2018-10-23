#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
import numpy.linalg as llg
from scipy.optimize import minimize
import sys

BOUNDS = {'lesson_gain_vec': (0.0001, 1.0), 'lesson_prerequisites_vec': (0.0001, 1.0)}
OPTIONS = {'maxiter': 100, 'disp': True, 'gtol': 1e-16, 'maxls': 20}


class LSEModel:
    def __init__(self):
        pass

    def sigmoid(self, z):
        '''
        sigmoid函数
        '''
        s = 1 / (1 + np.exp(-z))
        return s

    def vec_delta(self, dots, norm_q):
        '''
        计算向量差
        '''
        delta = dots / norm_q - norm_q
        return delta

    def prob(self, vec_delta):
        '''
        计算概率
        '''
        p = self.sigmoid(vec_delta)
        return p

    def lesson_gain_likelihood_fun(self, student_vec, lesson_gain_vec, lesson_prerequisites_vec, student_vec1, sigma):
        '''
        课程增益似然函数
        :param student_vec:
        :param lesson_gain_vec:
        :param lesson_prerequisites_vec:
        :param student_vec1:
        :param sigma:
        :return:
        '''
        dots = np.dot(student_vec, lesson_prerequisites_vec)
        norm_q = llg.norm(lesson_prerequisites_vec, axis=0).reshape(1, -1)
        delta = self.vec_delta(dots, norm_q)
        probability = self.prob(delta)

        mu = np.transpose(student_vec.reshape(student_vec.shape[0], student_vec.shape[1], 1), axes=[0, 2, 1]) \
             + np.einsum('ij,kj->ijk', probability, lesson_gain_vec)

        sigma_matrix = np.diag((np.ones(student_vec.shape[1]) * sigma ** 2).tolist())

        likelihood = -np.sum(
            -0.5 * np.einsum('ijk,ljk->ilk', np.einsum('ijk,kk->ijk', student_vec1 - mu, llg.inv(sigma_matrix)),
                             student_vec1 - mu)
            + np.log(1 / (np.sqrt((2 * np.pi) ** student_vec.shape[1] * llg.det(sigma_matrix))))
        )

        return likelihood

    def stretch(self, tensors):
        '''
        编码函数，将多个矩阵拼接成一个向量
        '''
        _dict = OrderedDict()
        cursor = 0
        for var_name, var_value in tensors.items():
            _dict[var_name] = {
                'index_start': cursor,
                'index_end': cursor + var_value.size,
                'shape': var_value.shape
            }
            cursor += var_value.size
        big_vec = np.concatenate([var_value.flat for var_value in tensors.values()])
        return big_vec, _dict

    def shrink(self, x, _dict):
        '''
        解码函数，将向量恢复成多个矩阵
        '''
        lesson_gain_vec, lesson_prerequisites_vec = (
            x[_dict[var_name]['index_start']:_dict[var_name]['index_end']].reshape(_dict[var_name]['shape']) for
            var_name in _dict)
        return lesson_gain_vec, lesson_prerequisites_vec

    def auxiliary(self, x, student_vec, student_vec1, sigma, _dict):
        '''
        辅助函数，minimize使用
        '''
        lesson_gain_vec, lesson_prerequisites_vec = self.shrink(x, _dict)
        lost = self.lesson_gain_likelihood_fun(
            student_vec, lesson_gain_vec, lesson_prerequisites_vec, student_vec1, sigma=sigma)
        return lost

    def set_bounds(self, xlen, _dict, bounds):
        '''
        设置边界
        :param xlen:
        :param _dict:
        :param bounds:
        :return:
        '''
        low_bound = np.zeros(xlen)
        up_bound = np.zeros(xlen)
        for varname, varvalue in _dict.items():
            low_bound = [bounds[varname][0] if _dict[varname]['index_start'] <= tmp <= _dict[varname]['index_end']
                         else low_bound[tmp] for tmp in range(len(low_bound))]
            up_bound = [bounds[varname][1] if _dict[varname]['index_start'] <= tmp <= _dict[varname]['index_end']
                        else up_bound[tmp] for tmp in range(len(up_bound))]
        low_bound = [None if bnd == -np.inf else bnd for bnd in low_bound]
        up_bound = [None if bnd == np.inf else bnd for bnd in up_bound]
        bound = list(zip(low_bound, up_bound))
        return bound

    def init_params(self, num_lessons, dimension):

        lesson_gain_vec = np.random.random((dimension, num_lessons))
        lesson_prerequisites_vec = np.random.random((dimension, num_lessons))

        x0, _dict = self.stretch(OrderedDict([('lesson_gain_vec', lesson_gain_vec),
                                              ('lesson_prerequisites_vec', lesson_prerequisites_vec)]))
        return x0, _dict

    def fit(self, dimension, student_vec, student_vec1, maxite=3, sigma=1):
        '''
        模型训练
        :param dimension:
        :param maxite:
        :param lbd:
        :return:
        '''
        num_lessons = student_vec1.shape[1]
        x0, _dict = self.init_params(num_lessons, dimension)  # 随机初始化参数
        bound = self.set_bounds(len(x0), _dict, BOUNDS)  # 设置参数边界

        result = minimize(self.auxiliary, x0.reshape(-1), args=(student_vec, student_vec1, sigma, _dict), bounds=bound,
                          method='L-BFGS-B', options=OPTIONS, jac=False)

        print('result', result.fun)
        print(self.shrink(result.x, _dict=_dict)[0].T)
        print(self.shrink(result.x, _dict=_dict)[1].T)


def test():
    lsee = LSEModel()
    sigma = 1
    k = 10  # 学生能力维度
    m = 30  # 学生数
    n = 1  # 题目数
    fogells = np.random.uniform(0.1, 1.0, (m, k))
    mclovins = np.random.uniform(0.1, 1.0, (m, k))
    seths = np.random.uniform(0.1, 1.0, (m, k))
    evans = np.random.uniform(0.1, 1.0, (m, k))

    s = np.concatenate((fogells, mclovins, seths, evans), axis=0)  # 学生当前能力

    l = np.random.uniform(0.0001, 0.5, (k, m))
    q = np.random.uniform(0.0001, 0.5, (k, m))
    # print('real l', l.T)
    # print('real q', q.T)

    dots = np.dot(s, q)
    norm_q = llg.norm(q, axis=0).reshape(1, -1)
    delta = lsee.vec_delta(dots, norm_q)
    probability = lsee.prob(delta)  # 计算学生获得课程增益的百分比

    s3d = s.reshape(s.shape[0], s.shape[1], 1)
    # s1 = s + np.dot(probability, l.T)  # 学生增长后的能力
    s1 = np.transpose(s3d, axes=[0, 2, 1]) + np.einsum('ij,kj->ijk', probability, l)  # 学生增长后的能力

    lsee.fit(k, s, s1, sigma=sigma)  # 训练模型


if __name__ == '__main__':
    test()
