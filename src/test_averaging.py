import pandas as pd
import numpy as np
import data_export


"""
 Interesting result! This mix improved score from 1022 to 1016!!!
"""

file1 = '../submission-ann/submission_ann-(82,84,73.5,75).csv'
file2 = '../submission-ann/submission_73-avg-5.csv'
file3 = '../submission-ann/submission_75-avg.csv'
file4 = '../submission-ann/submission_ann_(73-75.5)-avg.csv'
# file5 = '../submission-xgb/submission_xgb-30-avg.csv'
# file6 = '../submission-xgb/submission_xgb-31-avg.csv'
# file7 = '../submission-xgb/submission_xgb-35-avg.csv'
# file8 = '../submission-xgb/submission_xgb-38-avg.csv'
# file9 = '../submission-xgb/submission_xgb-40-avg-thresh-2000.csv'
# file10 = '../submission/submission_82-avg.csv'
# file11 = '../submission/submission_ann_(73-75.5)+xgb19-(0.55,0.45).csv'
# file12 = '../submission/submission_83-avg.csv'
file13 = '../submission-xgb/submission_xgb-43-avg.csv'
file14 = '../submission-ann/submission_93-avg.csv'
file15 = '../submission-ann/submission_95-avg.csv'

dt1 = pd.read_csv(file1)
dt2 = pd.read_csv(file2)
dt3 = pd.read_csv(file3)
dt4 = pd.read_csv(file4)
# dt5 = pd.read_csv(file5)
# dt6 = pd.read_csv(file6)
# dt7 = pd.read_csv(file7)
# dt8 = pd.read_csv(file8)
# dt9 = pd.read_csv(file9)
# dt10 = pd.read_csv(file10)
# dt11 = pd.read_csv(file11)
# dt12 = pd.read_csv(file12)
dt13 = pd.read_csv(file13)
dt14 = pd.read_csv(file14)
dt15 = pd.read_csv(file15)

for val1, val2 in zip(dt1['id'], dt13['id']):
    if val1 != val2:
        print 'Incorrect order'
        exit(0)

a1 = 0.45
a2 = 0.33
a3 = 0.34
a4 = 0.45
a9 = 0.5
a10 = 0.33
a11 = 0.5
a12 = 0.2
a13 = 0.55

loss_fin = 0.43 * ((dt2['loss'].values + dt3['loss'].values + dt14['loss'].values + dt15['loss'].values) / 4.0) + \
           0.57 * dt13['loss'].values
# loss_fin = (a1*dt1['loss'].values + a2*dt2['loss'].values + a3*dt3['loss'].values + a4*dt4['loss'].values)
# loss_fin = (a2*dt2['loss'].values + a3*dt3['loss'].values + a10*dt10['loss'].values)
# loss_fin = (a4*dt4['loss'].values + a13*dt13['loss'].values)
# loss_fin = (dt1['loss'].values +
#             dt2['loss'].values +
#             dt3['loss'].values +
#             dt10['loss'].values +
#             dt12['loss'].values) / 5.0

id_fin = dt1['id'].values

# print np.where(loss_fin > 30000)

data_export.save_submission_mix(id_fin, loss_fin, 'xgb43+ann(73.5,75,93,95)-(57,43)')
