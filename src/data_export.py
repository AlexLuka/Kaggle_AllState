import pandas as pd


def save_submission_ann(id_number, predicted, submission_number):
    submission = pd.DataFrame()
    submission['id'] = id_number
    submission['loss'] = predicted
    submission.to_csv('../submission-ann/submission_{}.csv'.format(submission_number), index=False)


def save_submission_xgb(id_number, predicted, submission_number):
    submission = pd.DataFrame()
    submission['id'] = id_number
    submission['loss'] = predicted
    submission.to_csv('../submission-xgb/submission_{}.csv'.format(submission_number), index=False)


def save_submission_mix(id_number, predicted, submission_number):
    submission = pd.DataFrame()
    submission['id'] = id_number
    submission['loss'] = predicted
    submission.to_csv('../submission-mix/submission_{}.csv'.format(submission_number), index=False)


def save_submission_rf(id_number, predicted, submission_number):
    submission = pd.DataFrame()
    submission['id'] = id_number
    submission['loss'] = predicted
    submission.to_csv('../submission-rf/submission_{}.csv'.format(submission_number), index=False)


def save_submission_knn(id_number, predicted, k_number, submission_number):
    submission = pd.DataFrame()
    submission['id'] = id_number
    submission['loss'] = predicted
    submission.to_csv('../submission/submission_{}_{}.csv'.format(submission_number, k_number), index=False)
