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


# ====================== Actual run ===================================
# performing tuning here
# important params:
shift_in = 220          # possible values are 200,220
shift_out = 170         # possible values are 100,200,220 for 220 in.
# make_submission = True
epochs = 50
n_sim = 15
split1 = 0.20067
run = 81

# train data
print 'Loading training data'
train_id, train_data, train_loss = src.data_reader.train_data_reader2()

# shifts_in = [10, 20, 30, 40, 60, 80, 100, 130, 150, 200, 220]
# dropouts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# err_test = []
# err_train = []

print 'Loading test data'
test_id, test_data = src.data_reader.test_data_reader2()

predict_res = np.zeros(shape=(test_id.shape[0], 1), dtype=float)


for ggg in range(n_sim):
    print '{0} Iteration {1} {0}'.format('=' * 40, ggg)

    tr_loss = np.log(train_loss + shift_in)
    mu = np.mean(tr_loss)
    sd = np.std(tr_loss)
    tr_loss = (tr_loss - mu) / sd

    print 'Split train data to train and cv sets'
    x_tr, x_tst, y_tr, y_tst = train_test_split(train_data,
                                                tr_loss,
                                                test_size=split1,  # make train array of size 256*n
                                                random_state=np.random.randint(np.iinfo(np.int32).max))
    print 'Shape train: {} x {}'.format(x_tr.shape[0], x_tr.shape[1])
    print 'Shape test: {} x {}'.format(x_tst.shape[0], x_tst.shape[1])

    m = nn_model(x_tr.shape[1],
                 400, 200, 200, 25,
                 dropout1=0.5,
                 dropout2=0.15,  # any value > 0.3
                 dropout3=0.15,
                 dropout4=0.1)

    m.fit_generator(generator=batch_generator(x_tr, y_tr, 128, True),
                    nb_epoch=epochs,
                    samples_per_epoch=x_tr.shape[0],
                    verbose=0
                    )

    print 'Prediction'
    predict_cv = m.predict_generator(generator=batch_generator_predict(x_tst, 1000), val_samples=x_tst.shape[0])
    predict_train = m.predict_generator(generator=batch_generator_predict(x_tr, 1000), val_samples=x_tr.shape[0])
    print 'Predicted {} train examples and {} test examples'.format(predict_train.shape[0], predict_cv.shape[0])

    # input: (actual, predicted)
    mae_test = mean_absolute_error(np.exp(sd*y_tst+mu)-shift_out, np.exp(sd*predict_cv+mu)-shift_out)
    mae_train = mean_absolute_error(np.exp(sd*y_tr+mu)-shift_out, np.exp(sd*predict_train+mu)-shift_out)

    print 'MAE test:  {}'.format(mae_test)
    print 'MAE train: {}'.format(mae_train)

    del (tr_loss, x_tr, x_tst, y_tr, y_tst)

    # predict on test data
    predict_test = m.predict_generator(generator=batch_generator_predict(test_data, 1000),
                                       val_samples=test_data.shape[0])

    # data_export.save_submission(test_id, np.exp(sd*predict_test+mu)-shift_out, '{}-{}'.format(run, ggg))
    predict_res += np.exp(sd * predict_test + mu) - shift_out

predict_res /= float(n_sim)

src.data_export.save_submission(test_id, predict_res, '{}-avg'.format(run))

# plt.plot(dropouts, err_test, 'or', label='test')
# plt.plot(dropouts, err_train, 'ob', label='train')
# plt.legend()
# plt.show()
