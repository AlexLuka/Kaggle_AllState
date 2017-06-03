import src.data_reader
import numpy as np
import matplotlib.pyplot as plt


train_id, train_data, train_loss = src.data_reader.train_data_reader()
test_id, test_data = src.data_reader.test_data_reader()

x = train_data[:, -10].toarray().T[0]
y = train_loss

ind1 = np.where(train_data[:, 93].toarray().T[0] == 1)[0]
ind2 = np.where(train_data[:, 495].toarray().T[0] == 1)[0]

ind1_test = np.where(test_data[:, 93].toarray().T[0] == 1)[0]
ind2_test = np.where(test_data[:, 495].toarray().T[0] == 1)[0]

print ind1_test
print ind2_test

for val1 in ind1:
    if val1 in ind2:
        print 'ID ={}, Loss = {}'.format(train_id[val1], train_loss[val1])

for val1 in ind1_test:
    if val1 in ind2_test:
        print 'ID ={}, Loss = {}'.format(test_id[val1], 0)

# for i in [93, 495]:
#     c = reduce(lambda a, b: a + b, train_data[:, i].toarray().T[0])
#     print 'i = {}, val_train = {}, val_test = {}'.format(i,
#                                                          c,
#                                                          reduce(lambda x, y: x + y, test_data[:, i].toarray().T[0])
#                                                          )
#     if c < 30:
#
#         plt.scatter(x, y, c=(train_data[:, i].toarray().T[0]*0.1))
#         plt.show()


    # c = np.corrcoef(train_data[:, 0].toarray().T, train_data[:, i].toarray().T)
    # print 'i ={}, corr = {}'.format(i, c[0, 1])

# print reduce(lambda x, y: x+y, train_data[:, 59].toarray().T[0])
