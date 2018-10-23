#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
import numpy.linalg as llg
from scipy.optimize import minimize


class LSEModel:

    def init_params(self, m, n, k, mu=1.0):
        '''
        初始化参数
        '''
        s = np.random.normal(mu, 1.5, (m, k))
        s[s < 0.0001] = 0.0001
        a = np.random.normal(mu, 1.5, (k, n))
        a[a < 0.0001] = 0.0001
        gm_s = np.random.normal(0, 0.16, (m, 1))
        gm_q = np.random.normal(0, 0.16, (1, n))

        return s, a, gm_s, gm_q

    def sigmoid(self, z):
        '''
        sigmoid函数
        '''
        s = 1 / (1 + np.exp(-z))
        return s

    def vec_delta(self, s, a, gm_s, gm_q, lbd=0.0001):
        '''
        计算向量差
        '''
        norm_a = llg.norm(a, axis=0).reshape(1, -1)
        # delta = np.dot(s, a) / (norm_a + lbd) - norm_a + gm_q + gm_s
        delta = np.dot(s, a) / norm_a - norm_a + gm_q + gm_s
        return delta

    def prob(self, s, a, gm_s, gm_q):
        '''
        计算概率
        '''
        p = self.sigmoid(self.vec_delta(s, a, gm_s=gm_s, gm_q=gm_q))
        return p

    def rev_likehood(self, s, a, y, gm_s, gm_q, lbd=0.01, with_gradient=False):
        '''
        负似然函数，优化时需要求它的极小值
        '''
        dots = np.dot(s, a)
        norm_a = llg.norm(a, axis=0).reshape(1, -1)
        delta = dots / norm_a - norm_a + gm_q + gm_s
        p = self.sigmoid(delta)
        # 调用计算函数

        l = - np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + \
            lbd * (np.sum(a * a) + np.sum(s * s))

        if not with_gradient:
            return l
        # caculate gradient    
        delta_y_p = y - p
        gradient_s = -np.dot(delta_y_p, (a / norm_a).T) + (2 * lbd) * s
        gradient_gm_s = -(delta_y_p).sum(axis=1).reshape(-1, 1)
        gradient_gm_q = -(delta_y_p).sum(axis=0).reshape(1, -1)
        # caculate gradient for a,need array shape like (s,q,k)
        num_students, num_questions = dots.shape
        embedding_dimension = s.shape[1]
        dots_3 = dots.reshape(num_students, num_questions, 1)
        norm_3 = norm_a.reshape(1, num_questions, 1)
        s_3 = s.reshape(num_students, 1, embedding_dimension)
        a_3 = a.T.reshape(1, num_questions, embedding_dimension)
        gradient_a = -np.einsum('sq,sqk->kq', delta_y_p, s_3 / norm_3 - (dots_3 / (norm_3 ** 2) + 1) * a_3 / norm_3) + (
                    2 * lbd) * a
        return l, gradient_s, gradient_a, gradient_gm_s, gradient_gm_q

    def gradient(self, s, a, gms, gmq, y, lbd):
        '''
        获取梯度
        '''
        norm_s = llg.norm(s, axis=1).reshape((s.shape[0], 1))
        norm_a = llg.norm(a, axis=0).reshape((1, a.shape[1]))
        p = self.prob(s=s, a=a, gm_s=gms, gm_q=gmq)
        ds = self.gradient_s(s, a, gms, gmq, y, p, lbd, norm_s, norm_a)
        da = self.gradient_a(s, a, gms, gmq, y, p, lbd, norm_s, norm_a)
        dgms = self.gradient_gms(s, a, gms, gmq, y, p, lbd, norm_s, norm_a)
        dgmq = self.gradient_gmq(s, a, gms, gmq, y, p, lbd, norm_s, norm_a)
        gradient_vec, gradient_dict = self.stretch(ds, da, dgms, dgmq)
        return gradient_vec.reshape(-1), gradient_dict

    def auxiliary(self, x, y, lbd, _dict):
        '''
        辅助函数，minimize使用
        '''
        s, a, gm_s, gm_q = self.shrink_all(x, _dict)

        lost, gradient_s, gradient_a, gradient_gms, gradient_gm_q = self.rev_likehood(s, a, y, gm_s=gm_s, gm_q=gm_q,
                                                                                      lbd=lbd, with_gradient=True)
        gradient_vec, gradient_dict = self.stretch2(OrderedDict([('gradient_s', gradient_s),
                                                                 ('gradient_a', gradient_a),
                                                                 ('gradient_gms', gradient_gms),
                                                                 ('gradient_gm_q', gradient_gm_q)]))
        return lost, gradient_vec

    def auxiliary_s(self, x, a, gm_s, gm_q, y, lbd, _dict):
        '''
        辅助函数求s
        '''
        s = self.shrink_s(x, _dict)
        result = self.rev_likehood(s, a, y, gm_s=gm_s, gm_q=gm_q, lbd=lbd)
        return result

    def shrink_all(self, x, _dict):
        '''
        解码函数，将向量恢复成多个矩阵
        '''

        s, a, gm_s, gm_q = (
        x[_dict[var_name]['index_start']:_dict[var_name]['index_end']].reshape(_dict[var_name]['shape']) for var_name in
        _dict)
        return s, a, gm_s, gm_q

    def shrink_s(self, x, _dict):
        '''
        只求s解码函数
        '''
        s = x[_dict['s']['index_start']: _dict['s']['index_end']].reshape(_dict['s']['shape'])
        return s

    def stretch(self, **args):
        '''
        编码函数，将多个矩阵拼接成一个向量
        '''
        _dict = OrderedDict()
        cursor = 0
        big_vec = None
        for key in args.keys():
            var = args[key]
            if big_vec == None:
                big_vec = var.reshape(-1)
            else:
                big_vec = np.hstack((big_vec, var.reshape(-1)))
            _dict[key] = {
                'index_start': cursor,
                'index_end': cursor + var.size,
                'shape': var.shape
            }
            cursor = _dict[key]['index_end']
        return big_vec, _dict

    def stretch2(self, tensors):
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

    def set_bounds_all(self, xlen, _dict, slow, sup, alow, aup, gslow, gsup, gqlow, gqup):
        '''
        设置参数的边界
        '''
        low_bound = np.zeros(xlen)
        up_bound = np.zeros(xlen)
        low_bound[_dict['s']['index_start']: _dict['s']['index_end']] = slow
        up_bound[_dict['s']['index_start']: _dict['s']['index_end']] = sup

        low_bound[_dict['a']['index_start']: _dict['a']['index_end']] = alow
        up_bound[_dict['a']['index_start']: _dict['a']['index_end']] = aup

        low_bound[_dict['gm_s']['index_start']: _dict['gm_s']['index_end']] = gslow
        up_bound[_dict['gm_s']['index_start']: _dict['gm_s']['index_end']] = gsup

        low_bound[_dict['gm_q']['index_start']: _dict['gm_q']['index_end']] = gqlow
        up_bound[_dict['gm_q']['index_start']: _dict['gm_q']['index_end']] = gqup

        low_bound = low_bound.tolist()
        up_bound = up_bound.tolist()

        low_bound = [None if bnd == -np.inf else bnd for bnd in low_bound]
        up_bound = [None if bnd == np.inf else bnd for bnd in up_bound]

        bound = zip(low_bound, up_bound)
        return bound

    def set_bounds_s(self, xlen, _dict, slow, sup):
        '''
        s边界设置函数
        '''
        low_bound = np.ones(xlen) * slow
        up_bound = np.ones(xlen) * sup

        low_bound = low_bound.tolist()
        up_bound = up_bound.tolist()

        low_bound = [None if bnd == -np.inf else bnd for bnd in low_bound]
        up_bound = [None if bnd == np.inf else bnd for bnd in up_bound]

        bound = zip(low_bound, up_bound)
        return bound

    def run(self, s, a, gm_s, gm_q, y, options, bounds, lbd=0.001):
        # s, a, gm_s, gm_q = self.init_params(m, n, k, mu=mu)
        x0, _dict = self.stretch(s=s, a=a, gm_s=gm_s, gm_q=gm_q)
        bound = self.set_bounds_all(len(x0), _dict, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
                                    bounds[6], bounds[7])
        result = minimize(self.auxiliary, x0.reshape(-1), args=(y, lbd, _dict), bounds=bound,
                          method='L-BFGS-B', options=options)
        ss, aa, gms, gmq = self.shrink_all(result.x, _dict=_dict)
        return ss, aa, gms, gmq

    def get_ability(self, s, a, gm_s, gm_q, y, options, bounds, lbd=0.001):
        x0, _dict = self.stretch(s=s)
        bound = self.set_bounds_s(len(x0), _dict, bounds[0], bounds[1])
        result = minimize(self.auxiliary_s, x0.reshape(-1), jac=True, args=(a, gm_s, gm_q, y, lbd, _dict), bounds=bound,
                          method='L-BFGS-B', options=options)
        ss = self.shrink_s(result.x, _dict)
        return ss

    def test_gradient(self):
        num_students = 2
        num_questions = 2
        embedding_dimension = 2
        s = np.random.random((num_students, embedding_dimension))
        a = np.random.random((embedding_dimension, num_questions))
        gms = np.random.random((num_students, 1))
        gmq = np.random.random((1, num_questions))
        s[:] = 1
        a[:] = 1
        gms[:] = 0
        gmq[:] = 0
        p = self.prob(s, a, gms, gmq)
        print('p', p)
        y = np.random.random(p.shape) < p
        print('y', y)
        x, _dict = self.stretch2(OrderedDict([('s', s), ('a', a), ('gms', gms), ('gmq', gmq)]))
        # print _dict
        cost, gradient = self.auxiliary(x, y, 0, _dict)
        num_gradient = np.zeros(x.shape)
        delta_x = 1e-13
        for i in range(x.shape[0]):
            x[i] += delta_x
            num_gradient[i] = (self.auxiliary(x, y, 0, _dict)[0] - cost) / delta_x
            x[i] -= delta_x
        print('gradient by function: ', gradient)
        print('gradient by num: ', num_gradient)
        print('error: ', np.abs(gradient - num_gradient))
        return gradient, num_gradient


if __name__ == '__main__':
    y = np.loadtxt('answer.txt')
    lse = LSEModel()
    ss, aa, gms, gmq = lse.run(10, 100, 2, y, lbd=0.001, options={'disp': 1, 'maxiter': 100})

    from sklearn.model_selection import train_test_split

    train, test = train_test_split()
