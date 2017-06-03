from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
import src.data_reader
import src.data_export
import xgboost as xgb
import numpy as np
from time import time


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds-labels
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess


# model params
shift_in = 220
shift_out = 180
run = 33
number_of_rounds = 1000
params = {
    # 'objective': 'reg:linear',
    'learning_rate': 0.04,
    'min_child_weight': 115,
    'eta': 0.3,
    'colsample_bytree': 0.6,
    'max_depth': 12,
    'subsample': 0.8,
    'alpha': 1,
    'gamma': 0.8,
    'silent': 1,
    'seed': 29537994,
    'booster': 'gbtree'
}

# train data
print 'Loading training data'
train_id, train_data, train_loss = src.data_reader.train_data_reader()

# transform loss to gaussian-like distribution with mean 0 and std 1
tr_loss = np.log(train_loss + shift_in)
mu = np.mean(tr_loss)
sd = np.std(tr_loss)
tr_loss = (tr_loss - mu) / sd


def eval_error(p, dt):
    labels = dt.get_label()
    mae = mean_absolute_error(np.exp(sd*p+mu)-shift_out, np.exp(sd*labels+mu)-shift_out)
    return 'mae', mae


print 'Loading test data'
test_id, test_data = src.data_reader.test_data_reader()
predict_test = np.zeros(test_id.shape[0], dtype=float)

# split train set to train and cv
print 'Split train set'
X_train, X_test, y_train, y_cv = train_test_split(train_data, tr_loss, test_size=0.2, random_state=0)

d_train = xgb.DMatrix(X_train, label=y_train)
d_cv = xgb.DMatrix(X_test, label=y_cv)
d_test = xgb.DMatrix(test_data)

n_rounds = 1

predict_train = np.zeros(y_train.shape[0], dtype=float)
predict_cv = np.zeros(y_cv.shape[0], dtype=float)

evallist = [(d_cv, 'eval'), (d_train, 'train')]

for v in range(n_rounds):
    t0 = time()
    print '{0} Iteration {1} {0}'.format('='*80, v)

    params['seed'] = np.random.randint(np.iinfo(np.int32).max)

    # the model
    print 'Train the model: number of rounds = {}'.format(number_of_rounds)
    model = xgb.train(params,
                      d_train,
                      number_of_rounds,
                      evallist,
                      feval=eval_error,
                      obj=logregobj,
                      early_stopping_rounds=25)

    # prediction
    print 'Make predictions'
    p1 = model.predict(d_train, ntree_limit=model.best_ntree_limit)
    p2 = model.predict(d_cv, ntree_limit=model.best_ntree_limit)
    p3 = model.predict(d_test, ntree_limit=model.best_ntree_limit)

    predict_train += np.exp(sd*p1+mu)-shift_out
    predict_cv += np.exp(sd*p2+mu)-shift_out
    predict_test += np.exp(sd*p3+mu)-shift_out

    print '\tMAE=({}, {})'.format(mean_absolute_error(np.exp(sd * y_train + mu) - shift_out,
                                                      np.exp(sd * p1 + mu) - shift_out),
                                  mean_absolute_error(np.exp(sd * y_cv + mu) - shift_out,
                                                      np.exp(sd * p2 + mu) - shift_out)
                                  )

    print 'simulation time: {} minutes'.format((time() - t0)/60.0)

predict_train /= float(n_rounds)
predict_cv /= float(n_rounds)
predict_test /= float(n_rounds)

mae_train = mean_absolute_error(np.exp(sd*y_train+mu)-shift_out, predict_train)
mae_cv = mean_absolute_error(np.exp(sd*y_cv+mu)-shift_out, predict_cv)

print '\n\nTOTAL MAE (tr,cv): ( {} , {} )'.format(mae_train, mae_cv)

src.data_export.save_submission_xgb(test_id, predict_test, 'xgb-{}-avg'.format(run))

print 'Done'
