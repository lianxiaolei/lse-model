import numpy as np
import pymc as pymc
import matplotlib.pyplot as plt

n_items = 100
n_stu = 100
# real_a = np.random.lognormal(0,0.2,(n_items,))
real_b1 = np.random.normal(0, 1, (1, n_items))
real_theta1 = np.random.normal(0, 1, (n_stu, 1))
real_b2 = np.random.normal(0, 1, (1, n_items))
real_theta2 = np.random.normal(0, 1, (n_stu, 1))
# uv=np.random.random((n_items,n_stu))<c + (1.0-c) / (1.0 + np.exp(-D*real_a*(real_theta-real_b)))
uv = np.random.random((n_items, n_stu)) < 1.0 / (1.0 + np.exp(-(real_theta1 - real_b1))) / (
    1.0 + np.exp(-(real_theta2 - real_b2)))
mask = np.random.random(uv.shape) < 0.7


def absm(a, b):
    return np.abs(a - b).mean()


def uirt(uv, mask):
    n_stu, n_items = uv.shape

    # 难度参数
    b1 = pymc.Normal('b1', mu=0, tau=1, value=np.zeros((1, n_items,)))
    # 能力参数
    theta1 = pymc.Normal('theta1', mu=0, tau=1, value=np.zeros((n_stu, 1)))
    # 难度参数
    b2 = pymc.Normal('b2', mu=0, tau=1, value=np.zeros((1, n_items,)))
    # 能力参数
    theta2 = pymc.Normal('theta2', mu=0, tau=1, value=np.zeros((n_stu, 1)))

    # 区分度参数
    #     a=pymc.Lognormal('a',mu=0,tau=25,value=np.ones((n_items,)))



    @pymc.deterministic
    def sigmoid(theta1=theta1, b1=b1, theta2=theta2, b2=b2):
        return 1.0 / (1.0 + np.exp(-(theta1 - b1))) / (1.0 + np.exp(-(theta2 - b2)))

    # 答题矩阵
    u = pymc.Bernoulli('u', p=sigmoid, value=uv, observed=True)

    params = [b1, theta1, b2, theta2, sigmoid]
    params.append(u)

    m = pymc.MCMC(params)
    print u.logp
    m.sample(100000, 40000, 12)
    print u.logp
    eb1 = np.array([b1.stats()['mean']])
    et1 = np.array([theta1.stats()['mean']]).T
    eb2 = np.array([b2.stats()['mean']])
    et2 = np.array([theta2.stats()['mean']]).T
    m.stats()

    print absm(eb1.reshape(real_b1.shape), real_b1), absm(et1.reshape(real_theta1.shape), real_theta1)
    print absm(eb2.reshape(real_b2.shape), real_b2), absm(et2.reshape(real_theta2.shape), real_theta2)
    return eb1, et1, eb2, et2


eb1, et1, eb2, et2 = uirt(uv, mask)
k = 0.5
uv_test = np.ones((n_items, n_stu)) * k < 1.0 / (1.0 + np.exp(-(et1 - eb1))) / (1.0 + np.exp(-(et2 - eb2)))
result = uv_test == uv
print result.sum() / n_items / n_stu

from sklearn.metrics import roc_curve

uv_list = np.array([int(uvv) for uvv in uv.flat]) + 1
uv_test_list = np.array([int(uvv) for uvv in uv_test.flat])
uv_p = 1.0 / (1.0 + np.exp(-(et1 - eb1))) / (1.0 + np.exp(-(et2 - eb2)))
uv_p_list = np.array([uvv for uvv in uv_p.flat])

fpr, tpr, thresholds = roc_curve(uv_list, uv_p_list, pos_label=2)

from sklearn.metrics import auc

plt.plot(fpr, tpr, label=thresholds)
plt.plot([0, 1], [1, 0])

print auc(fpr, tpr)

print fpr
print tpr
print thresholds
