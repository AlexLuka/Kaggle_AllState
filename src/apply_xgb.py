from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import data_reader
import data_export
import xgboost as xgb
import numpy as np


"""
 Wanted to start with XGB and just copy/pasted someone's kernel.
 Improved it a bit, and got quite good results.

"""


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds-labels
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess


# model params
shift_in = 200
shift_out = 200
run = 27
number_of_rounds = 1100
params = {
    # 'objective': 'reg:linear',
    'learning_rate': 0.03,
    'min_child_weight': 100,
    'eta': 0.03,
    'colsample_bytree': 0.7,
    'max_depth': 12,
    'subsample': 0.95,
    'alpha': 1,
    'gamma': 1,
    'silent': 1,
    'seed': 29537994,
    'early_stopping_rounds': 25,
    'booster': 'gbtree'
}

# train data
print 'Loading training data'
train_id, train_data, train_loss = data_reader.train_data_reader2()

# transform loss to gaussian-like distribution with mean 0 and std 1
tr_loss = np.log(train_loss + shift_in)
mu = np.mean(tr_loss)
sd = np.std(tr_loss)
tr_loss = (tr_loss - mu) / sd


def eval_error(p, dt):
    labels = dt.get_label()
    return 'mae: ', mean_absolute_error(np.exp(sd*p+mu)-shift_out, np.exp(sd*labels+mu)-shift_out)
    # return 'mae: ', mean_absolute_error(np.exp(p)-shift_out, np.exp(labels)-shift_out)


print 'Loading test data'
test_id, test_data = data_reader.test_data_reader2()
# predict_res = np.zeros(shape=(test_id.shape[0], 1), dtype=float)
d_test = xgb.DMatrix(test_data)
predict_test = np.zeros(test_id.shape[0], dtype=float)

n_runs = 5
n_rounds = 4

for kor in range(n_rounds):
    # split train set to train and cv
    print 'Split train set'
    X_train, X_test, y_train, y_cv = train_test_split(train_data, tr_loss, test_size=0.2, random_state=0)

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_cv = xgb.DMatrix(X_test)

    predict_train = np.zeros(y_train.shape[0], dtype=float)
    predict_cv = np.zeros(y_cv.shape[0], dtype=float)

    for v in range(n_runs):
        print '{0} Iteration {1} Round {2} {0}'.format('='*80, v, kor)

        params['seed'] = np.random.randint(np.iinfo(np.int32).max)

        # the model
        print 'Train the model: number of rounds = {}'.format(number_of_rounds)
        model = xgb.train(params,
                          d_train,
                          number_of_rounds,
                          obj=logregobj,
                          feval=eval_error)

        # prediction
        print 'Make predictions'
        p1 = model.predict(d_train)
        p2 = model.predict(d_cv)
        p3 = model.predict(d_test)

        predict_train += np.exp(sd*p1+mu)-shift_out
        predict_cv += np.exp(sd*p2+mu)-shift_out
        predict_test += np.exp(sd*p3+mu)-shift_out

        # predict_train += np.exp(p1) - shift_out
        # predict_cv += np.exp(p2) - shift_out
        # predict_test += np.exp(p3) - shift_out

        # print '\tMAE=({}, {})'.format(mean_absolute_error(np.exp(sd * y_train + mu) - shift_out,
        #                                                   np.exp(sd * p1 + mu) - shift_out),
        #                               mean_absolute_error(np.exp(sd * y_cv + mu) - shift_out,
        #                                                   np.exp(sd * p2 + mu) - shift_out)
        #                               )

    predict_train /= float(n_runs)
    predict_cv /= float(n_runs)

    print '\tMAE=({}, {})'.format(mean_absolute_error(np.exp(sd * y_train + mu) - shift_out,
                                                      predict_train),
                                  mean_absolute_error(np.exp(sd * y_cv + mu) - shift_out,
                                                      predict_cv)
                                  )

predict_test /= float(n_runs * n_rounds)

# mae_train = mean_absolute_error(np.exp(sd*y_train+mu)-shift_out, predict_train)
# mae_cv = mean_absolute_error(np.exp(sd*y_cv+mu)-shift_out, predict_cv)
# print '\n\nTOTAL MAE (tr,cv): ( {} , {} )'.format(mae_train, mae_cv)

data_export.save_submission_xgb(test_id, predict_test, 'xgb-{}-avg'.format(run))

print 'Done'
