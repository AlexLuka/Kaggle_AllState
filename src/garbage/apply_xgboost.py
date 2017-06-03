from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from src.data_reader import get_train_num, get_train_cat, get_train_loss, get_test_cat, get_test_num, get_test_id
import xgboost as xgb
import numpy as np
from sklearn.grid_search import GridSearchCV
import pandas as pd


def eval_error(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))

num_train = get_train_num()
los_train = get_train_loss()
#
X_train, X_test, y_train, y_test = train_test_split(num_train, los_train, test_size=0.35, random_state=0)

shift = 0
y = np.log(y_train)+shift
y_t = np.log(y_test)+shift
print y

d_train = xgb.DMatrix(X_train, label=y)
d_test = xgb.DMatrix(X_test)

param = {
    'min_child_weight': 1,
    'eta': 0.01,
    'colsample_bytree': 0.5,
    'max_depth': 12,
    'subsample': 0.8,
    'alpha': 1,
    'gamma': 1,
    'silent': 1,
    'verbose_eval': True,
    'seed': 29537994
}

ind_params = {
    'learning_rate': 0.1,
    'n_estimators': 300,
    'max_depth': 15,
    'min_child_weight': 1,
    'seed': 8728305,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
}

# Training XGB
# for n in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
#     # print 'Train XGB'
#     model = xgb.train(param, d_train, n, feval=eval_error)
#     prediction = np.exp(model.predict(d_test)-shift)
#
#     print 'n = {}, mea = {}'.format(n, mean_absolute_error(prediction, y_t))

cv_params = {'max_depth': [3, 5, 7, 9, 12, 18],
             'min_child_weight': [1, 2, 3, 4, 5]}
optimization = GridSearchCV(xgb.XGBRegressor(**ind_params),
                            cv_params, scoring='mean_absolute_error',
                            cv=5, n_jobs=-1)
optimization.fit(X_train, y)
for params in optimization.best_params_:
   print params
   #ind_params[params] = optimization.best_params[params]
exit(0)


# Get test data
num_test = get_test_num()
id_test = get_test_id()

# Train on full set
y = np.log(los_train)
full_train = xgb.DMatrix(num_train, label=y)

model = xgb.train(param, full_train, 1000, feval=eval_error)
predict_test = np.exp(model.predict(xgb.DMatrix(num_test))-shift)

print "{}  {}".format(len(predict_test), len(id_test))
submission = pd.DataFrame()
submission['id'] = id_test
submission['loss'] = predict_test
submission.to_csv('../submission/subm4.csv', index=False)
