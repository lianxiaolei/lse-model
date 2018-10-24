#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Created by Teacher Dong"""

from collections import OrderedDict
import numpy as np
import numpy.linalg as llg
from scipy.optimize import minimize

MIN_A = 0.0001
# BOUNDS = [0, np.inf, MIN_A, np.inf, -0.1, 0.1, -0.1, 0.1]
BOUNDS = {'student_vec': {'low': 0, 'up': np.inf}, 'question_vec': {'low': MIN_A, 'up': np.inf},
          'gm_s': {'low': -0.1, 'up': 0.1}, 'gm_q': {'low': -0.1, 'up': 0.1}}
STUDENT_BOUNDS = {'student_vec': {'low': 0, 'up': np.inf}, 'gm_s': {'low': -0.1, 'up': 0.1}}
OPTIONS = {'maxiter': 100}


class LSEModel:
    def __init__(self):
        self.est_s = None
        self.est_a = None
        self.est_gm_s = None
        self.est_gm_q = None
        self.likehood = None
        self.lbd = None

    def sigmoid(self, z):
        '''
        sigmoid函数
        '''
        s = 1 / (1 + np.exp(-z))
        return s

    def vec_delta(self, dots, norm_a, gm_s, gm_q):
        '''
        计算向量差
        '''
        delta = dots / norm_a - norm_a + gm_s + gm_q
        return delta

    def prob(self, vec_delta):
        '''
        计算概率
        '''
        p = self.sigmoid(vec_delta)
        return p

    def negative_log_likehood_cost(self, student_vec, question_vec, y, gm_s, gm_q, lbd=0.001, with_gradient=True):
        '''
        负对数似然目标函数，优化时需要求它的极小值
        '''
        dots = np.dot(student_vec, question_vec)
        norm_a = llg.norm(question_vec, axis=0).reshape(1, -1)
        delta = self.vec_delta(dots, norm_a, gm_s, gm_q)
        probability = self.prob(delta)
        # 调用计算函数
        # negative_log_likehood = - np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability)) + \
        #                         lbd * (np.sum(question_vec * question_vec) + np.sum(student_vec * student_vec))
        negative_log_likehood = - np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability)) + \
                                lbd * (np.sum(np.dot(question_vec, question_vec.T)) +
                                       np.sum(np.dot(student_vec, student_vec.T)))
        # print(negative_log_likehood)

        if not with_gradient:
            return negative_log_likehood
        # caculate gradient
        delta_y_p = y - probability

        # gradient_s = -np.dot(delta_y_p, probability.T).dot(1 - probability)
        # gradient_s = gradient_s.dot((question_vec / norm_a).T) + (2 * lbd) * student_vec
        gradient_s = -np.dot(delta_y_p, (question_vec / norm_a).T) + (2 * lbd) * student_vec  # dL/dθ 简化之后的结果

        gradient_gm_s = -(delta_y_p).sum(axis=1).reshape(-1, 1)
        gradient_gm_q = -(delta_y_p).sum(axis=0).reshape(1, -1)

        # caculate gradient for a,need array shape like (s,q,k)
        num_students, num_questions = dots.shape
        embedding_dimension = student_vec.shape[1]

        dots_3 = dots.reshape(num_students, num_questions, 1)
        norm_3 = norm_a.reshape(1, num_questions, 1)

        student_vec_3 = student_vec.reshape(num_students, 1, embedding_dimension)
        question_vec_3 = question_vec.T.reshape(1, num_questions, embedding_dimension)

        gradient_a = -np.einsum(
            'sq,sqk->kq', delta_y_p,
            student_vec_3 / norm_3 - (dots_3 / (norm_3 ** 2) + 1) * question_vec_3 / norm_3) + (2 * lbd) * question_vec

        return negative_log_likehood, gradient_s, gradient_a, gradient_gm_s, gradient_gm_q

    def auxiliary(self, x, y, lbd, _dict, with_gradient=True):
        '''
        辅助函数，minimize使用
        '''
        student_vec, question_vec, gm_s, gm_q = self.shrink_all(x, _dict)
        lost, gradient_s, gradient_a, gradient_gm_s, gradient_gm_q = \
            self.negative_log_likehood_cost(student_vec, question_vec, y,
                                            gm_s=gm_s,
                                            gm_q=gm_q,
                                            lbd=lbd,
                                            with_gradient=with_gradient)

        gradient_vec, gradient_dict = \
            self.stretch(OrderedDict([('gradient_s', gradient_s),
                                      ('gradient_a', gradient_a),
                                      ('gradient_gm_s', gradient_gm_s),
                                      ('gradient_gm_q', gradient_gm_q)]))

        print(gradient_s.shape)

        return lost, gradient_vec

    def auxiliary_s(self, x, question_vec, gm_q, y, lbd, _dict, with_gradient=True):
        '''
        辅助函数，minimize使用
        '''
        student_vec, gm_s, = self.shrink_all(x, _dict)
        lost, gradient_s, gradient_a, gradient_gm_s, gradient_gm_q = \
            self.negative_log_likehood_cost(student_vec, question_vec, y,
                                            gm_s=gm_s,
                                            gm_q=gm_q,
                                            lbd=lbd,
                                            with_gradient=with_gradient)

        gradient_vec, gradient_dict = \
            self.stretch(OrderedDict([('gradient_s', gradient_s),
                                      ('gradient_gm_s', gradient_gm_s)]))
        return lost, gradient_vec

    def stretch(self, tensors):
        '''
        编码函数，将多个矩阵拼接成一个向量
        '''
        _dict = OrderedDict()
        cursor = 0
        for var_name, var_value in tensors.items():
            print(var_name, var_value.shape, var_value.size)
            _dict[var_name] = {
                'index_start': cursor,
                'index_end': cursor + var_value.size,
                'shape': var_value.shape
            }
            cursor += var_value.size
        big_vec = np.concatenate([var_value.flat for var_value in tensors.values()])
        return big_vec, _dict

    def shrink_all(self, x, _dict):
        '''
        解码函数，将向量恢复成多个矩阵
        '''
        # print _dict, x
        student_vec, question_vec, gm_s, gm_q = (
            x[_dict[var_name]['index_start']:_dict[var_name]['index_end']].reshape(_dict[var_name]['shape']) for
            var_name in _dict)
        return student_vec, question_vec, gm_s, gm_q

    def set_bounds_all(self, xlen, _dict, bounds):
        low_bound = np.zeros(xlen)
        up_bound = np.zeros(xlen)
        for varname, varvalue in _dict.items():
            low_bound = [bounds[varname]['low'] if _dict[varname]['index_start'] <= tmp <= _dict[varname]['index_end']
                         else low_bound[tmp] for tmp in range(len(low_bound))]
            up_bound = [bounds[varname]['up'] if _dict[varname]['index_start'] <= tmp <= _dict[varname]['index_end']
                        else up_bound[tmp] for tmp in range(len(up_bound))]

        low_bound = [None if bnd == -np.inf else bnd for bnd in low_bound]
        up_bound = [None if bnd == np.inf else bnd for bnd in up_bound]

        bound = list(zip(low_bound, up_bound))

        return bound

    def init_params(self, num_students, num_questions, dimension):

        init_s = np.random.random((num_students, dimension))
        init_a = np.random.random((dimension, num_questions)) + MIN_A
        init_gm_s = np.random.random((num_students, 1)) * 2 - 1
        init_gm_q = np.random.random((1, num_questions)) * 2 - 1
        x0, _dict = self.stretch(OrderedDict([('student_vec', init_s),
                                              ('question_vec', init_a),
                                              ('gm_s', init_gm_s),
                                              ('gm_q', init_gm_q)]))

        return x0, _dict

    def fit(self, y, dimension, maxite=3, lbd=0.001):
        '''
        初始化参数
        '''
        self.lbd = lbd
        num_students = y.shape[0]
        num_questions = y.shape[1]

        best_negative_log_likehood = np.inf
        best_x = None
        for ite in range(maxite):
            x0, _dict = self.init_params(num_students, num_questions, dimension)
            bound = self.set_bounds_all(len(x0), _dict, BOUNDS)
            result = minimize(self.auxiliary, x0.reshape(-1), args=(y, lbd, _dict), bounds=bound,
                              method='L-BFGS-B', options=OPTIONS, jac=True)
            if result.fun < best_negative_log_likehood:
                best_negative_log_likehood = result.fun
                best_x = result.x
        self.est_s, self.est_a, self.est_gm_s, self.est_gm_q = self.shrink_all(best_x, _dict=_dict)
        # print self.est_s.shape, self.est_a.shape, self.est_gm_s.shape, self.est_gm_q.shape
        self.likehood = -best_negative_log_likehood + lbd * (np.sum(self.est_a * self.est_a) +
                                                             np.sum(self.est_s * self.est_s))
        # print(self.likehood)

    def predict(self, student_vec):

        dots = np.dot(student_vec, self.est_a)
        norm_a = llg.norm(self.est_a, axis=0).reshape(1, -1)
        vec_delta = self.vec_delta(dots, norm_a, self.est_gm_s, self.est_gm_q)
        prob = self.prob(vec_delta)
        return prob

    def measure(self, y, maxite):
        best_negative_log_likehood = np.inf
        best_x = None
        for ite in range(maxite):
            x0, _dict = self.stretch(OrderedDict([('student_vec', self.est_s),
                                                  ('gm_s', self.est_gm_s)]))
            bound = self.set_bounds_all(len(x0), _dict, STUDENT_BOUNDS)
            result = minimize(self.auxiliary_s, x0.reshape(-1), args=(self.est_a, self.est_gm_q, y, self.lbd, _dict),
                              bounds=bound, method='L-BFGS-B', options=OPTIONS, jac=True)
            if result.fun < best_negative_log_likehood:
                best_negative_log_likehood = result.fun
                best_x = result.x
        student_vec, gm_s = self.shrink_s(best_x, _dict)
        return student_vec, gm_s

    def save(self, est_s, est_a, est_gm_s, est_gm_q):
        np.savetxt('files/est_s.txt', est_s)
        np.savetxt('files/est_a.txt', est_a)
        np.savetxt('files/est_gm_s.txt', est_gm_s)
        np.savetxt('files/est_gm_q.txt', est_gm_q)

    def load(self):
        self.est_s = np.loadtxt('files/est_s.txt')
        self.est_a = np.loadtxt('files/est_a.txt')
        self.est_gm_s = np.loadtxt('files/est_gm_s.txt')
        self.est_gm_s = self.est_gm_s.reshape((len(self.est_gm_s), 1))
        self.est_gm_q = np.loadtxt('files/est_gm_q.txt')
        self.est_gm_q = self.est_gm_q.reshape((1, len(self.est_gm_q)))


if __name__ == '__main__':
    from LSE import LSEModel

    lse = LSEModel()

    m = 10
    n = 10
    k = 60
    s = np.random.normal(1.5, 1.5, (m, k))
    s[s < 0.0001] = 0.0001
    a = np.random.normal(1.5, 1.5, (k, n))
    a[a < 0.0001] = 0.0001
    gm_s = np.random.normal(0, 0.01, (m, 1))
    gm_q = np.random.normal(0, 0.01, (1, n))

    u = np.zeros((m, n))
    norm_a = llg.norm(a, axis=0).reshape(1, -1)
    dots = np.dot(s, a)
    delta = lse.vec_delta(dots, norm_a, gm_s, gm_q)

    p = lse.prob(delta)

    rand = np.random.random(p.shape)

    u[rand < p] = 1.0
    u[rand >= p] = 0.0

    lse.fit(u, 2)
