import data_reader
import matplotlib.pyplot as plt
import numpy as np


train_id, train_data, train_loss = data_reader.train_data_reader2()

print 'Data shape: ', train_data.shape

p = np.where(train_loss > 40000)

ind1 = 1076
ind2 = 1068
ind3 = 1069
ind4 = 1067

a1 = train_data[:, ind1].toarray()
a2 = train_data[:, ind2].toarray()
a3 = train_data[:, ind3].toarray()
a4 = train_data[:, ind4].toarray()

a1p = train_data[p, ind1].toarray()
a2p = train_data[p, ind2].toarray()
a3p = train_data[p, ind3].toarray()
a4p = train_data[p, ind4].toarray()

plt.plot(np.multiply(np.multiply(a1, a2), np.multiply(a3, a4)), a1, '.')

plt.plot(np.multiply(np.multiply(a1p, a2p), np.multiply(a3p, a4p)), a1p, '.r')

plt.show()
