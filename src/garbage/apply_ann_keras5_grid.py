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
import src.data_export
import matplotlib.pyplot as plt


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
# performing tuning here
# important params:
shift_in = 220          # possible values are 200,220
shift_out = 220         # possible values are 100,200,220 for 220 in.
make_submission = True
epochs = 70
split1 = 0.20067
runs = 5

# train data
print 'Loading training data'
train_id, train_data, train_loss = src.data_reader.train_data_reader()

print 'Loading test data'
test_id, test_data = src.data_reader.test_data_reader()

tr_loss = np.log(train_loss + shift_in)
mu = np.mean(tr_loss)
sd = np.std(tr_loss)
tr_loss = (tr_loss - mu) / sd

x_tr, x_tst, y_tr, y_tst = train_test_split(train_data,
                                            tr_loss,
                                            test_size=split1,  # make train array of size 256*n
                                            random_state=np.random.randint(np.iinfo(np.int32).max))

predict_train = np.zeros(shape=(y_tr.shape[0], 1), dtype=float)
predict_cv = np.zeros(shape=(y_tst.shape[0], 1), dtype=float)
predict_test = np.zeros(shape=(test_id.shape[0], 1), dtype=float)

for val in range(runs):
    print '{0} Iteration {1} {0}'.format('='*40, val)
    # tr_loss = np.log(train_loss + shift_in)
    # mu = np.mean(tr_loss)
    # sd = np.std(tr_loss)
    # tr_loss = (tr_loss - mu) / sd
    #
    # print 'Split train data to train and cv sets'
    # x_tr, x_tst, y_tr, y_tst = train_test_split(train_data,
    #                                             tr_loss,
    #                                             test_size=split1,  # make train array of size 256*n
    #                                             random_state=np.random.randint(np.iinfo(np.int32).max))
    # print 'Shape train: {} x {}'.format(x_tr.shape[0], x_tr.shape[1])
    # print 'Shape test: {} x {}'.format(x_tst.shape[0], x_tst.shape[1])

    m = nn_model(x_tr.shape[1],
                 800, 200, 50,
                 dropout1=0.7,
                 dropout2=0.4,
                 dropout3=0.2)

    m.fit_generator(generator=batch_generator(x_tr, y_tr, 256, True),
                    nb_epoch=epochs,
                    samples_per_epoch=x_tr.shape[0],
                    verbose=0
                    )

    print 'Prediction'
    p1 = m.predict_generator(generator=batch_generator_predict(x_tst, 1000), val_samples=x_tst.shape[0])
    p2 = m.predict_generator(generator=batch_generator_predict(x_tr, 1000), val_samples=x_tr.shape[0])
    p3 = m.predict_generator(generator=batch_generator_predict(test_data, 1000), val_samples=test_data.shape[0])

    print 'Predicted {} train examples and {} test examples'.format(predict_train.shape[0], predict_cv.shape[0])

    predict_cv += (np.exp(sd*p1+mu)-shift_out)
    predict_train += (np.exp(sd*p2+mu)-shift_out)
    predict_test += (np.exp(sd*p3+mu)-shift_out)

    print 'Local MAE (tst, tr)= ({}, {})'.format(
        mean_absolute_error(np.exp(sd * y_tst + mu) - shift_out, np.exp(sd * p1 + mu) - shift_out),
        mean_absolute_error(np.exp(sd * y_tr + mu) - shift_out, np.exp(sd * p2 + mu) - shift_out),
    )

# average result over 5 runs
predict_cv /= float(runs)
predict_train /= float(runs)
predict_test /= float(runs)

mae_cv = mean_absolute_error(np.exp(sd*y_tst+mu)-shift_out, predict_cv)
mae_train = mean_absolute_error(np.exp(sd*y_tr+mu)-shift_out, predict_train)

print 'MAE cv:  {}'.format(mae_cv)
print 'MAE train: {}'.format(mae_train)

# # predict on test data
if make_submission:
    src.data_export.save_submission(test_id, predict_test, '70-avg')
