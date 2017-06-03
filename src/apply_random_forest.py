import data_reader
import data_export
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from time import time


"""
The first attempt to work with RandomForest Regressor
"""


# params
shift = 220
split1 = 0.3

print 'Loading training data'
train_id, train_data, train_loss = data_reader.train_data_reader2()

print 'Loading test data'
test_id, test_data = data_reader.test_data_reader2()

tr_loss = np.log(train_loss + shift)
mu = np.mean(tr_loss)
sd = np.std(tr_loss)
tr_loss = (tr_loss - mu) / sd

print 'Split train data to train and cv sets'
x_tr, x_tst, y_tr, y_tst = train_test_split(train_data,
                                            tr_loss,
                                            test_size=split1,
                                            random_state=np.random.randint(np.iinfo(np.int32).max))
print 'Shape train: {} x {}'.format(x_tr.shape[0], x_tr.shape[1])
print 'Shape test: {} x {}'.format(x_tst.shape[0], x_tst.shape[1])

rfr = RandomForestRegressor(n_estimators=8,
                            criterion='mse',
                            max_features=0.9,
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=50,
                            bootstrap=True,
                            random_state=np.random.randint(np.iinfo(np.int32).max),
                            verbose=2,
                            oob_score=True,
                            warm_start=False,
                            n_jobs=-1)

t0 = time()
rfr.fit(x_tr, y_tr)
print 'Fit took {} minutes'.format((time() - t0) / 60.0)

p_cv = rfr.predict(x_tst)
p_tr = rfr.predict(x_tr)
p_test = rfr.predict(test_data)
print p_cv

mae_cv = mean_absolute_error(np.exp(sd * y_tst + mu) - shift, np.exp(sd * p_cv + mu) - shift)
mae_train = mean_absolute_error(np.exp(sd * y_tr + mu) - shift, np.exp(sd * p_tr + mu) - shift)

print 'MAE = ({}, {})'.format(mae_train, mae_cv)

# test data
data_export.save_submission_rf(test_id, np.exp(sd * p_test + mu) - shift, '3')
