import src.data_reader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


train_id, train_data, train_loss = src.data_reader.train_data_reader()
print train_data[:, 0:1176].shape

# plt.hist(train_data[:, 0:1176].toarray(), 100)
# plt.show()
# a, b, c = np.linalg.svd(train_data[:, 0:10].toarray())
# print '{}, {}, {}'.format(a.shape, b.shape, c.shape)

tsvd = TruncatedSVD(n_components=50)

tsvd.fit(train_data[:, 0:1176].T)
print tsvd.components_.T.shape

plt.hist(tsvd.components_.T[:, 40], 100)
plt.show()
