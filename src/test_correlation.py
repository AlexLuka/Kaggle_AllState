import numpy as np
import pandas as pd
import data_reader
import matplotlib.pyplot as plt


# dt1 = pd.read_csv('../submission/submission_75-avg.csv')
# dt2 = pd.read_csv('../submission-xgb/submission_xgb-19-avg.csv')
# dt3 = pd.read_csv('../submission/submission_72-avg.csv')
# dt4 = pd.read_csv('../submission-xgb/submission_xgb-14-avg.csv')
#
# cc = np.corrcoef(dt2['loss'].values, dt4['loss'].values)
#
# print cc

# train data
print 'Loading training data'
train_id, train_data, train_loss = data_reader.train_data_reader2()

# mark outliers as 1
loss_outliers = np.zeros(train_loss.shape)
loss_outliers[np.where(train_loss > 40000)] = 1
loss_outliers = np.asarray(loss_outliers, dtype=bool)
#
# loss_inliers = np.zeros(train_loss.shape)
# loss_inliers[np.where(train_loss <= 40000)] = 1
# loss_inliers = np.asarray(loss_inliers, dtype=bool)

# ttt = train_data[loss_inliers, :]
# lll = train_loss[loss_inliers]

# print loss_inliers

corr_val = []
loss_diff = []
corr_test = dict()

print 'Loading test data'
test_id, test_data = data_reader.test_data_reader2()

for out, l1 in zip(train_data[loss_outliers, :], train_loss[loss_outliers]):
    out = out.toarray()
    corr_test[l1] = []
    # for i, val in enumerate(train_data):
    #     val = val.toarray()
    #     corr_val.append(np.corrcoef(out, val)[0, 1])
    #     loss_diff.append(l1 - train_loss[i])
        # print 'HH'
    for vval in test_data:
        vval = vval.toarray()
        corr_test[l1].append(np.corrcoef(out, vval)[0, 1])

    print 'Value {} is done'.format(l1)

for key in corr_test.keys():
    # plt.hist(corr_val, 200, color='b')
    plt.hist(corr_test[key], 200, color='r')
    plt.title('Value: {}'.format(key))
    plt.show()
