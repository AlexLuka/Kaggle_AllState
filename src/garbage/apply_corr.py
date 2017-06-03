import src.data_reader
import numpy as np
import matplotlib.pyplot as plt
from time import time


"""
Have no idea what is going on here)))
"""


def get_bigauss_pars(example0, loss0, data_train, label_train):
    corr_dist = []
    loss_dist = []

    for ex, loss in zip(data_train, label_train):
        corr_dist.append(np.corrcoef(example0, ex)[0][1])
        loss_dist.append(loss0 - loss)

    mcorr = np.mean(corr_dist)
    mloss = np.mean(loss_dist)
    stdcorr = np.std(corr_dist)
    stdloss = np.std(loss_dist)
    corrcoef = np.corrcoef(corr_dist, loss_dist)[0][1]
    return mcorr, mloss, stdcorr, stdloss, corrcoef


def fi_cost(x, y, mu1, mu2, sig1, sig2, rho):
    return np.exp(-(((x-mu1)/sig1)**2 + ((y-mu2)/sig2)**2 - 2*rho*(x-mu1)*(y-mu2)/(sig1*sig2))/(2*(1-rho**2)))


def gi_deriv_y(y, mu2, sig1, sig2, rho):
    return -((y-mu2)/sig2**2 - rho/(sig1*sig2))/(1-rho**2)


num_train = src.data_reader.get_train_num()
cat_train = src.data_reader.get_train_cat()
los_train = np.log(src.data_reader.get_train_loss())

all_train = np.hstack((num_train, cat_train))

example = all_train[11]
loss = los_train[11]

test_example = all_train[100]
test_loss = los_train[100]

t = time()
mux, muy, sigx, sigy, rhoxy = get_bigauss_pars(example, loss, all_train, los_train)
print 'Gauss pars: {}'.format(t-time())


loss23 = np.random.normal(float(np.mean(los_train)), float(np.std(los_train)))
print 'Initial cost = {}, diff = {}'.format(loss23, loss-loss23)
corr_dist = np.corrcoef(example, test_example)[0][1]
lossold = loss-loss23
f = fi_cost(corr_dist, lossold, mux, muy, sigx, sigy, rhoxy)
print 'Cost function: {}'.format(f)
alp = 0.1

n = 0
while n<50:
    lossold += alp*f*gi_deriv_y(lossold, muy, sigx, sigy, rhoxy)
    f = fi_cost(corr_dist, lossold, mux, muy, sigx, sigy, rhoxy)
    print 'New loss = {}, new cost = {}'.format(lossold, f)
    n+=1

print 'New loss: {}, actual loss: {}'.format(lossold, loss-loss23)
# for ex, loss in zip(all_train, los_train):
#     corr_dist.append(np.corrcoef(example0, ex)[0][1])
#     loss_dist.append(loss0 - loss)
#
# plt.plot(corr_dist, loss_dist, 'o')
# plt.show()
#
# plt.hist2d(corr_dist, loss_dist, bins=100)
# plt.show()
#
# plt.hist(corr_dist, 200)
# plt.show()
#
# plt.hist(los_train, 200)
# plt.show()

