import pandas as pd
import numpy as np
import src.data_export


file13 = '../submission-xgb/submission_xgb-43-avg.csv'
file14 = '../submission-ann/submission_93-avg.csv'
file15 = '../submission-ann/submission_95-avg.csv'
file2 = '../submission-ann/submission_73-avg-5.csv'
file3 = '../submission-ann/submission_75-avg.csv'

dt13 = pd.read_csv(file13)
dt14 = pd.read_csv(file14)
dt15 = pd.read_csv(file15)
dt2 = pd.read_csv(file2)
dt3 = pd.read_csv(file3)

l1 = dt13['loss'].values
l2 = dt14['loss'].values
l3 = dt2['loss'].values
l4 = dt3['loss'].values
l5 = dt15['loss'].values

l_ann = (l2 + l3 + l4 + l5) / 4.0

l_res = np.zeros(shape=(l1.shape[0], ), dtype=float)
print np.corrcoef(l1, l_ann)

for i in range(l1.shape[0]):
    # print '{:>10f} {:>10f} {:>4.3f}'.format(l1[i], l2[i], l1[i] - l2[i])

    marge = np.abs(l1[i] - l_ann[i])

    if marge > 2000:
        l_res[i] = 0.4 * l1[i] + 0.6 * l_ann[i]         # max still may work
        print 'Wide margin: {}'.format(marge)
    else:
        l_res[i] = 0.55*l1[i] + 0.45*l_ann[i]

src.data_export.save_submission_mix(dt13['id'].values, l_res, 'xgb43+ann(73,75,93,95)-marga3')
