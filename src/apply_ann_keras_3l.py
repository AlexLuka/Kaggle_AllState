import numpy as np
import data_reader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
import data_export
import os


# ====================== Batch generators ============================
def batch_generator(x, y, batch_size, shuffle):
    # chenglong code for fitting from generator
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
             n1, n2, n3,
             dropout1=0.4,
             dropout2=0.4,
             dropout3=0.2):
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

    model.add(Dense(1, init='he_normal', activation='linear'))

    model.compile(loss='mae', optimizer='adadelta')
    return model


# ====================== Actual run ===================================
def main_f():
    # performing tuning here
    # important params:
    shift_in = 200          # possible values are 200,220
    shift_out = 200         # possible values are 100,200,220 for 220 in.
    # make_submission = True
    epochs = 50
    n_folds = 10
    split1 = 0.20067
    run = 90

    # train data
    print 'Loading training data'
    train_id, train_data, train_loss = data_reader.train_data_reader7()

    print 'Loading test data'
    test_id, test_data = data_reader.test_data_reader7()

    predict_res = np.zeros(shape=(test_id.shape[0], 1), dtype=float)

    tr_loss = np.log(train_loss + shift_in)
    mu = np.mean(tr_loss)
    sd = np.std(tr_loss)
    tr_loss = (tr_loss - mu) / sd

    subdir = '../model-ann-{}'.format(run)
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=np.random.randint(np.iinfo(np.int32).max)).split(train_data)

    # for ggg in range(n_folds):
    for i, (train_index, val_index) in enumerate(kf):
        print '{0} Iteration {1} {0}'.format('=' * 40, i)

        x_tr, x_val = train_data[train_index, :], train_data[val_index, :]
        y_tr, y_val = tr_loss[train_index], tr_loss[val_index]

        print 'Shape train: {} x {}'.format(x_tr.shape[0], x_tr.shape[1])
        print 'Shape valid: {} x {}'.format(x_val.shape[0], x_val.shape[1])

        m = nn_model(x_tr.shape[1],
                     800, 200, 50,
                     dropout1=0.45,
                     dropout2=0.2,
                     dropout3=0.15)

        m.fit_generator(generator=batch_generator(x_tr, y_tr, 128, True),
                        nb_epoch=epochs,
                        samples_per_epoch=x_tr.shape[0],
                        verbose=2
                        )

        print 'Prediction'
        predict_val = m.predict_generator(generator=batch_generator_predict(x_val, 1000), val_samples=x_val.shape[0])
        predict_train = m.predict_generator(generator=batch_generator_predict(x_tr, 1000), val_samples=x_tr.shape[0])
        print 'Predicted {} train examples and {} test examples'.format(predict_train.shape[0], predict_val.shape[0])

        # input: (actual, predicted)
        mae_val = mean_absolute_error(np.exp(sd*y_val+mu)-shift_out, np.exp(sd*predict_val+mu)-shift_out)
        mae_train = mean_absolute_error(np.exp(sd*y_tr+mu)-shift_out, np.exp(sd*predict_train+mu)-shift_out)

        print 'MAE test:  {}'.format(mae_val)
        print 'MAE train: {}'.format(mae_train)

        # predict on test data
        predict_test = m.predict_generator(generator=batch_generator_predict(test_data, 1000),
                                           val_samples=test_data.shape[0])

        # data_export.save_submission(test_id, np.exp(sd*predict_test+mu)-shift_out, '{}-{}'.format(run, ggg))
        predict_res += np.exp(sd * predict_test + mu) - shift_out

        m.save(os.path.join(subdir, 'fold-{}.h5'.format(i)))

    predict_res /= float(n_folds)

    data_export.save_submission_ann(test_id, predict_res, '{}-avg'.format(run))


if __name__ == '__main__':
    main_f()
