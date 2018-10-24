# coding:utf8

import numpy as np
import scipy.linalg as llg
import scipy as sp

MIN_A = 0.0001
# BOUNDS = [0, None, MIN_A, None, -0.1, 0.1, -0.1, 0.1]
BOUNDS = {'student_vec': {'low': 0, 'up': None}, 'question_vec': {'low': MIN_A, 'up': None},
          'gm_s': {'low': -0.1, 'up': 0.1}, 'gm_q': {'low': -0.1, 'up': 0.1}}
STUDENT_BOUNDS = {'student_vec': {'low': 0, 'up': None}, 'gm_s': {'low': -0.1, 'up': 0.1}}
OPTIONS = {'maxiter': 100}


class Lse(object):
    def __init__(self):
        self.s = None
        self.a = None
        self.b = None
        self.c = None
        self.gms = None
        self.gma = None

    @staticmethod
    def calc_norm(x, axis=0):
        """

        :param x:
        :param axis:
        :return:
        """
        return llg.norm(x, axis=axis)

    @staticmethod
    def sigmoid(x):
        """

        :param x:
        :return:
        """
        if x == 0:
            x = 1e-6
        return 1 / (1 + np.exp(-x))

    def calc_delta(self, s, a, gms, gma):
        """

        :param s:
        :param a:
        :param gms:
        :param gma:
        :return:
        """
        return s.dot(a) / self.calc_norm(a, axis=0) \
            - self.calc_norm(a, axis=0) + gms + gma

    def _calc_delta_vec(self):
        """

        :return:
        """
        self.s.dot(self.a) / self.calc_norm(self.a, axis=0) \
            - self.calc_norm(self.a, axis=0) + self.gms + self.gma

    def calc_prob(self, delta_vec):
        """

        :param delta_vec:
        :return:
        """
        R = self.sigmoid(delta_vec)
        return R

    def neg_log_likelihood_cost(self, student_vec, question_vec, y, gm_s, gm_q, lbd=1e-3, with_gradient=False):
        dots = student_vec.dot(question_vec)
        norm_a = self.calc_norm(question_vec, axis=0)

        delta = self.calc_delta(student_vec, question_vec, gm_s, gm_q)
        R = self.calc_prob(delta)

        negative_log_likehood = - y * np.log(R) + (1 - y) * np.log(1 - R) + \
              lbd * (np.sum(student_vec.dot(student_vec.T)) +
                     np.sum(student_vec.dot(student_vec.T)))  # add the L2 regularzation

        if not with_gradient:
            return negative_log_likehood

        error = y - R
        gradient_s = error * question_vec / norm_a + 2 * lbd * student_vec

