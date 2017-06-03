import src.data_reader
import matplotlib.pyplot as plt
import numpy as np
import src.data_reader
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.optimizers import Adadelta
from scipy.sparse import csr_matrix
import src.data_export


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
             hid1,
             hid2,
             hid3,
             dropout1=0.4,
             dropout2=0.4,
             dropout3=0.2):
    model = Sequential()

    model.add(Dense(hid1, input_dim=input_size, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout1))

    model.add(Dense(hid2, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout2))

    model.add(Dense(hid3, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout3))

    model.add(Dense(1, init='he_normal', activation='linear'))

    model.compile(loss='mae', optimizer='adadelta')
    return model


def run_model(data_train, loss_train, data_test,
              hid1,
              hid2,
              hid3,
              shift_in=200,
              shift_out=200,
              epochs=50,
              split1=0.20067,
              dropout1=0.3,
              dropout2=0.3,
              dropout3=0.25,
              batch_size=256):
    # shift_in = 200  # possible values are 200,220
    # shift_out = 200  # possible values are 100,200,220 for 220 in.
    # make_submission = True
    # epochs = 50
    # split1 = 0.20067

    tr_loss = np.log(loss_train + shift_in)

    err_test = []
    err_train = []

    print 'Split train data to train and cv sets'
    x_tr, x_tst, y_tr, y_tst = train_test_split(data_train,
                                                tr_loss,
                                                test_size=split1,  # make train array of size 256*n
                                                random_state=np.random.randint(np.iinfo(np.int32).max))

    print 'Shape train: {} x {}'.format(x_tr.shape[0], x_tr.shape[1])
    print 'Shape test: {} x {}'.format(x_tst.shape[0], x_tst.shape[1])

    m = nn_model(x_tr.shape[1],
                 hid1,
                 hid2,
                 hid3,
                 dropout1=dropout1,
                 dropout2=dropout2,  # any value > 0.3
                 dropout3=dropout3)

    m.fit_generator(generator=batch_generator(x_tr, y_tr, batch_size, True),
                    nb_epoch=epochs,
                    samples_per_epoch=x_tr.shape[0],
                    verbose=1
                    )

    print 'Prediction'
    predict_cv = m.predict_generator(generator=batch_generator_predict(x_tst, 1000), val_samples=x_tst.shape[0])
    predict_train = m.predict_generator(generator=batch_generator_predict(x_tr, 1000), val_samples=x_tr.shape[0])
    print 'Predicted {} train examples and {} test examples'.format(predict_train.shape[0], predict_cv.shape[0])

    # input: (actual, predicted)
    mae_test = mean_absolute_error(np.exp(y_tst)-shift_out, np.exp(predict_cv)-shift_out)
    mae_train = mean_absolute_error(np.exp(y_tr)-shift_out, np.exp(predict_train)-shift_out)

    print 'MAE test:  {}'.format(mae_test)
    print 'MAE train: {}'.format(mae_train)

    err_train.append(mae_train)
    err_test.append(mae_test)

    del (tr_loss, x_tr, x_tst, y_tr, y_tst)

    # predict on test data
    predict_test = m.predict_generator(generator=batch_generator_predict(data_test, 1000),
                                       val_samples=data_test.shape[0])

    return np.exp(predict_test)-shift_out, mae_test, mae_train

# ====================== Actual run ===================================
# performing tuning here
# important params:
# shift_in = 200          # possible values are 200,220
# shift_out = 200         # possible values are 100,200,220 for 220 in.
# make_submission = True
# epochs = 50
#
# split1 = 0.20067

# train data
print 'Loading train and test data'
train_id, train_data, train_loss = src.data_reader.train_data_reader()
test_id, test_data = src.data_reader.test_data_reader()

train_gr1 = np.where(train_data[:, -10].toarray() < -0.2)[0]
train_gr2 = np.where(train_data[:, -10].toarray() >= -0.2)[0]

test_gr1 = np.where(test_data[:, -10].toarray() < -0.2)[0]
test_gr2 = np.where(test_data[:, -10].toarray() >= -0.2)[0]

print 'GR1={}, GR2={}'.format(train_data[train_gr1, :].shape, train_data[train_gr2, :].shape)

# group1
train_data1 = train_data[train_gr1, :]
train_id1 = train_id[train_gr1]
train_loss1 = train_loss[train_gr1]
train_data1[:, -10] = (train_data1[:, -10].toarray() - np.mean(train_data1[:, -10].toarray())) / \
                      np.std(train_data1[:, -10].toarray())

test_id1 = test_id[test_gr1]
test_data1 = test_data[test_gr1]

# group2
train_data2 = train_data[train_gr2, :]
train_id2 = train_id[train_gr2]
train_loss2 = train_loss[train_gr2]
train_data2[:, -10] = (train_data2[:, -10].toarray() - np.mean(train_data2[:, -10].toarray())) / \
                      np.std(train_data2[:, -10].toarray())

test_id2 = test_id[test_gr2]
test_data2 = test_data[test_gr2]

dropouts = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

err_tst = []
err_tr = []

for i, dr in enumerate(dropouts):
    print 'Iteration: {}\tDropout = {}'.format(i, dr)
    # run model on group 1
    predict_test1, er1, er2 = run_model(train_data1, train_loss1, test_data1,
                                        800, 200, 50,
                                        shift_in=220,
                                        shift_out=200,
                                        epochs=150,
                                        split1=0.10097,
                                        dropout1=dr,
                                        dropout2=0.4,
                                        dropout3=0.2,
                                        batch_size=512)
    err_tst.append(er1)
    err_tr.append(er2)

    predict_test2, _, _ = run_model(train_data2, train_loss2, test_data2,
                                    800, 300, 50,
                                    shift_in=200,
                                    shift_out=200,
                                    epochs=150,
                                    split1=0.10493,
                                    dropout1=0.5,
                                    dropout2=0.4,
                                    dropout3=0.2,
                                    batch_size=256)

    predicted_value = np.vstack((predict_test1, predict_test2))
    predicted_id = np.hstack((test_id1, test_id2))

    src.data_export.save_submission(predicted_id, predicted_value, 39 + i)

# plt.hist(train_data[:, -10].toarray(), 100)
# plt.show()
plt.plot(dropouts, err_tst, 'ok', label='test')
plt.plot(dropouts, err_tr, 'ob', label='train')
plt.legend()
plt.show()
