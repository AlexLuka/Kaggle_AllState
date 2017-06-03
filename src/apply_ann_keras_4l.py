import numpy as np
import data_reader
import data_export
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


# ====================== Batch generators ============================
def batch_generator(x, y, batch_size, shuffle):
    # chenglong code for fiting from generator
    # (https://www.kaggle.com/c/talkingdata-mobile-
    # user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(x.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(x.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        x_batch = x[batch_index, :].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield x_batch, y_batch
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generator_predict(x, batch_size):
    number_of_batches = x.shape[0] / np.ceil(x.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(x.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        x_batch = x[batch_index, :].toarray()
        counter += 1
        yield x_batch
        if counter == number_of_batches:
            counter = 0


# ====================== ANN model ===============================
def nn_model(input_size,
             n1, n2, n3, n4,
             dropout1=0.4,
             dropout2=0.4,
             dropout3=0.2,
             dropout4=0.1):
    model = Sequential()

    model.add(Dense(n1, input_dim=input_size, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout1))

    model.add(Dense(n2, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout2))

    model.add(Dense(n3, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout3))

    model.add(Dense(n4, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout4))

    model.add(Dense(1, init='he_normal', activation='linear'))

    model.compile(loss='mae', optimizer='adadelta')
    return model


def main_f():
    # important params:
    shift_in = 220              # possible values are 200,220
    shift_out = 170             # possible values are 100,200,220 for 220 in.
    save_submissions = True     # whether to save each fold or not
    epochs = 40
    n_folds = 10
    n_bags = 5
    split1 = 0.20067
    run = 95

    # Load test and train data
    print 'Loading training data'
    train_id, train_data, train_loss = data_reader.train_data_reader2()
    print 'Loading test data'
    test_id, test_data = data_reader.test_data_reader2()

    # perform transformation of the target value
    # (sigma, mu) transformation gives about +5 LB points
    tr_loss = np.log(train_loss + shift_in)
    mu = np.mean(tr_loss)
    sd = np.std(tr_loss)
    tr_loss = (tr_loss - mu) / sd

    # array to sum up predictions
    predict_res = np.zeros(shape=(test_id.shape[0], 1), dtype=float)

    # folds
    kf = KFold(n_splits=n_folds,
               shuffle=True,
               random_state=np.random.randint(np.iinfo(np.int32).max)).split(train_data)

    # if we want to keep all intermediate results, create a separate directory for all submissions
    subdir = '../model-ann-{}'.format(run)
    if save_submissions and not os.path.exists(subdir):
        print 'Create submission directory'
        os.mkdir(subdir)

    for i, (train_index, test_index) in enumerate(kf):
        # NOTE! If you use this method to split train set,
        # generate train test of size = batch_size * n, where n is an integer
        # Otherwise, you will get a warning message:
        #   >> Epoch comprised more than `samples_per_epoch` samples,
        #   >> which might affect learning results. Set `samples_per_epoch`
        #   >> correctly to avoid this warning.
        #
        # Not sure how appearance of that message may affect on the final result.
        # You have been warned!
        # print 'Split train data to train and cv sets'
        # x_tr, x_val, y_tr, y_val = train_test_split(train_data,
        #                                             tr_loss,
        #                                             test_size=split1,  # make train array of size 256*n
        #                                             random_state=np.random.randint(np.iinfo(np.int32).max))

        # If you are using KFold, the size of train
        x_tr, x_val = train_data[train_index, :], train_data[test_index, :]
        y_tr, y_val = tr_loss[train_index], tr_loss[test_index]

        for bag in range(n_bags):
            print '{0} Fold {1} Bag {2} {0}'.format('=' * 40, i, bag)

            m = nn_model(x_tr.shape[1],
                         300, 200, 200, 25,
                         dropout1=0.4,
                         dropout2=0.14,
                         dropout3=0.14,
                         dropout4=0.1)

            m.fit_generator(generator=batch_generator(x_tr, y_tr, 128, True),
                            nb_epoch=epochs,
                            samples_per_epoch=x_tr.shape[0],
                            verbose=2
                            )

            print '\tPrediction'
            predict_cv = m.predict_generator(generator=batch_generator_predict(x_val, 1000), val_samples=x_val.shape[0])
            predict_train = m.predict_generator(generator=batch_generator_predict(x_tr, 1000), val_samples=x_tr.shape[0])

            # input: (actual, predicted)
            mae_test = mean_absolute_error(np.exp(sd*y_val+mu)-shift_out, np.exp(sd*predict_cv+mu)-shift_out)
            mae_train = mean_absolute_error(np.exp(sd*y_tr+mu)-shift_out, np.exp(sd*predict_train+mu)-shift_out)

            print '\tMAE (valid, train):  ({}, {})'.format(mae_test, mae_train)

            # predict on test data
            predict_test = m.predict_generator(generator=batch_generator_predict(test_data, 1000),
                                               val_samples=test_data.shape[0])

            #
            predict_res += np.exp(sd * predict_test + mu) - shift_out

            if save_submissions:
                m.save('{}/fold-{}-{}.h5'.format(subdir, i, bag))

    predict_res /= float(n_folds*n_bags)
    data_export.save_submission_ann(test_id, predict_res, '{}-avg'.format(run))


if __name__ == '__main__':
    main_f()
