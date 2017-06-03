from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
import src.data_reader
import src.data_export
import xgboost as xgb
import numpy as np


# model params
shift_in = 220
shift_out = 200
run = 12
number_of_rounds = 2000
params = {
    'objective': 'reg:linear',
    'learning_rate': 0.03,
    'min_child_weight': 120,
    'eta': 0.03,
    'colsample_bytree': 0.6,
    'max_depth': 12,
    'subsample': 0.8,
    'alpha': 1,
    'gamma': 1,
    'silent': 0,
    'seed': 29537994,
    'early_stopping_rounds': 25
}

# train data
print 'Loading training data'
train_id, train_data, train_loss = src.data_reader.train_data_reader()

# transform loss to gaussian-like distribution with mean 0 and std 1
tr_loss = np.log(train_loss + shift_in)
mu = np.mean(tr_loss)
sd = np.std(tr_loss)
tr_loss = (tr_loss - mu) / sd


def eval_error(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae: ', mean_absolute_error(np.exp(sd*preds+mu)-shift_out, np.exp(sd*labels+mu)-shift_out)


# shifts_in = [10, 20, 30, 40, 60, 80, 100, 130, 150, 200, 220]
# dropouts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
err_test = []
err_train = []

print 'Loading test data'
test_id, test_data = src.data_reader.test_data_reader()
predict_res = np.zeros(shape=(test_id.shape[0], 1), dtype=float)

# split train set to train and cv
print 'Split train set'
X_train, X_test, y_train, y_test = train_test_split(train_data, tr_loss, test_size=0.2, random_state=0)

d_train = xgb.DMatrix(X_train, label=y_train)
d_cv = xgb.DMatrix(X_test)
d_test = xgb.DMatrix(test_data)

# the model
print 'Train the model: number of rounds = {}'.format(number_of_rounds)
model = xgb.train(params,
                  d_train,
                  number_of_rounds,
                  feval=eval_error)

# prediction
print 'Make predictions'
predict_train = model.predict(d_train)
predict_cv = model.predict(d_cv)
predict_test = model.predict(d_test)

mae_train = mean_absolute_error(np.exp(sd*y_train+mu)-shift_out, np.exp(sd*predict_train+mu)-shift_out)
mae_cv = mean_absolute_error(np.exp(sd*y_test+mu)-shift_out, np.exp(sd*predict_cv+mu)-shift_out)

print 'MAE (tr,cv): ( {} , {} )'.format(mae_train, mae_cv)

src.data_export.save_submission(test_id, np.exp(sd * predict_test + mu) - shift_out, 'xgb-{}'.format(run))

print 'Done'

# # Train on full set
# y = np.log(los_train)
# full_train = xgb.DMatrix(num_train, label=y)
#
# model = xgb.train(param, full_train, 1000, feval=eval_error)
# predict_test = np.exp(model.predict(xgb.DMatrix(num_test))-shift)
#
# print "{}  {}".format(len(predict_test), len(id_test))
# submission = pd.DataFrame()
# submission['id'] = id_test
# submission['loss'] = predict_test
# submission.to_csv('../submission/subm4.csv', index=False)
