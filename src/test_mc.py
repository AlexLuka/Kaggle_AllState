import pandas as pd
import xgboost as xgb
import numpy as np
import data_export
from time import time
from sklearn.metrics import mean_absolute_error


"""
    Monte-Carlo search of weights for XGB model with
    combinations of categorical features
"""
shift = 200


def get_train_data():
    ddt = pd.read_pickle('../data5/data-train')
    data_id = ddt['id'].values
    data_loss = ddt['loss'].values
    ddt = ddt.drop(['id', 'loss'], axis=1)
    return data_id, data_loss, ddt.values


def get_test_data():
    ddt = pd.read_pickle('../data5/data-test')
    data_id = ddt['id'].values
    ddt = ddt.drop(['id', 'loss'], axis=1)
    return data_id, ddt.values


def test_the_model(sim_n, n_folds_test, threshold=2000):
    test_id, test_data = get_test_data()
    predict_loss = np.zeros(test_id.shape[0], dtype=float)
    d_test = xgb.DMatrix(test_data)

    #
    average_counter = 0
    for i in range(n_folds_test):

        clf = xgb.Booster()
        clf.load_model('../xgb_models/run-1/model-{}'.format(i))

        print 'Iteration: {}    best score: {}'.format(i, clf.attributes()['best_score'])
        # print clf.attributes()
        if float(clf.attributes()['best_score']) < threshold:
            p = clf.predict(d_test)
            predict_loss += (np.exp(p) - shift)
            average_counter += 1

    predict_loss /= float(average_counter)
    data_export.save_submission_xgb(test_id, predict_loss, 'xgb-{}-avg-thresh-{}'.format(sim_n, threshold))
    print 'Perform average of {} simulation out of {} with threshold {}'.format(average_counter,
                                                                                n_folds_test,
                                                                                threshold)


def load_models(n_folds_test):
    models_list = []

    for i in range(n_folds_test):
        clf = xgb.Booster()
        clf.load_model('../xgb_models/run-1/model-{}'.format(i))
        models_list.append(clf)

    return models_list


def predictions(models_, d_matrix, ex):
    predicted_loss = np.zeros(shape=(len(models_), ex), dtype=float)

    for var, model_ in enumerate(models_):
        p = model_.predict(d_matrix)
        predicted_loss[var, :] = (np.exp(p) - shift)

    return predicted_loss


def energy(predicted_loss, weights_, true_loss):
    predicted_weighted_sum = np.average(predicted_loss, axis=0, weights=weights_)
    return mean_absolute_error(true_loss, predicted_weighted_sum)


if __name__ == '__main__':
    t0 = time()

    number_of_folds = 26
    sigma = 0.1
    sweeps = 2000

    # first of all, load all existing models
    models = load_models(number_of_folds)

    # get the test and train data
    print 'Load the train data'
    train_id, train_loss, train_data = get_train_data()
    d_train = xgb.DMatrix(train_data)
    del train_data

    # initialize default weights. We can choose them to be random and normalized,
    # or all weights can be equal. I prefer the latter option
    print 'Init weights'
    weights = np.ones(number_of_folds, dtype=float) / float(number_of_folds)

    # matrix of size: number_of_folds x number_of_train_examples
    print 'Calculate predictions for train set'
    predicted_loss_train = predictions(models, d_train, train_id.shape[0])
    del d_train         # we don't need it anymore

    print 'Calculate the energy of the model'
    e = energy(predicted_loss_train, weights, train_loss)

    for sweep in range(sweeps):                  # sweeps
        for j in range(number_of_folds):    # update each weight
            # change j-th weight
            w = weights
            w[j] = np.random.normal(loc=weights[j], scale=(sigma/float(number_of_folds)))

            e_new = energy(predicted_loss_train, w, train_loss)

            # accept if this is better value
            if e_new < e:
                e = e_new
                weights = w
        print 'Sweep {}: energy = {}'.format(sweep, e)
        # print weights

    del predicted_loss_train
    print 'MC takes {} minutes'.format((time() - t0)/60.0)

    print 'Make prediction'
    test_id, test_data = get_test_data()
    d_test = xgb.DMatrix(test_data)
    del test_data

    predicted_loss_test = predictions(models, d_test, test_id.shape[0])
    predicted_weighted_test = np.average(predicted_loss_test, axis=0, weights=weights)

    data_export.save_submission_xgb(test_id, predicted_weighted_test, 'xgb-mc-avg-({},{})'.format(sigma, sweeps))
