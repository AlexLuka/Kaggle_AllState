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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


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


def reduce_data(arr, arr2, new_components=50):
    # apply truncated singular value decomposition in order ot reduce categorical data
    t_svd = TruncatedSVD(n_components=new_components)

    print 'shapes: {} {}'.format(arr[:, :1176].T.shape, arr2[:, :1176].T.shape)

    arr_reduced_cat = t_svd.fit_transform(arr[:, :1176].T).T

    print arr_reduced_cat.shape

    # arr_reduced_cat = t_svd.components_.T
    arr_reduced_cat2 = t_svd.transform(arr2[:, :1176].T)

    print arr_reduced_cat2

    sc = StandardScaler()
    a = sc.fit_transform(arr_reduced_cat)
    b = sc.transform(arr_reduced_cat2)

    # convert sparse matrix to numpy array
    arr_reduced_num = arr[:, 1176:].toarray()
    arr_reduced_num2 = arr2[:, 1176:].toarray()

    return np.hstack((a, arr_reduced_num)), np.hstack((b, arr_reduced_num2))


# ====================== Actual run ===================================
# performing tuning here
# important params:
shift_in = 220          # possible values are 200,220
shift_out = 200         # possible values are 100,200,220 for 220 in.
make_submission = True
epochs = 50
n_important_cat = 100
split1 = 0.20067

# train data
print 'Loading training data'
train_id, train_data, train_loss = src.data_reader.train_data_reader()
test_id, test_data = src.data_reader.test_data_reader()

train_data, test_data = reduce_data(train_data, test_data, new_components=n_important_cat)

# shifts_in = [10, 20, 30, 40, 60, 80, 100, 130, 150, 200, 220]
dropouts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
err_test = []
err_train = []

for dr in [1]:
    tr_loss = np.log(train_loss + shift_in)

    print 'Split train data to train and cv sets'
    x_tr, x_tst, y_tr, y_tst = train_test_split(train_data,
                                                tr_loss,
                                                test_size=split1,  # make train array of size 256*n
                                                random_state=np.random.randint(np.iinfo(np.int32).max))
    print 'Shape train: {} x {}'.format(x_tr.shape[0], x_tr.shape[1])
    print 'Shape test: {} x {}'.format(x_tst.shape[0], x_tst.shape[1])

    m = nn_model(x_tr.shape[1],
                 400, 200, 10,
                 dropout1=0.5,
                 dropout2=0.3,
                 dropout3=0.2)

    m.fit(x_tr, y_tr,
          batch_size=256,
          nb_epoch=50,
          verbose=1,
          shuffle=True,
          validation_data=(x_tst, y_tst))
    # m.fit_generator(generator=batch_generator(x_tr, y_tr, 256, True),
    #                 nb_epoch=epochs,
    #                 samples_per_epoch=x_tr.shape[0],
    #                 verbose=1
    #                 )

    print 'Prediction'
    predict_train = m.predict(x_tr, batch_size=1024)
    predict_cv = m.predict(x_tst, batch_size=1024)
    # predict_cv = m.predict_generator(generator=batch_generator_predict(x_tst, 1000), val_samples=x_tst.shape[0])
    # predict_train = m.predict_generator(generator=batch_generator_predict(x_tr, 1000), val_samples=x_tr.shape[0])
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
    if make_submission:
        # test_id, test_data = data_reader.test_data_reader()
        # predict_test = m.predict_generator(generator=batch_generator_predict(test_data, 1000),
        #                                    val_samples=test_data.shape[0])
        predict_test = m.predict(test_data, 1000)
        src.data_export.save_submission(test_id, np.exp(predict_test) - shift_out, 44)

# plt.plot(dropouts, err_test, 'or', label='test')
# plt.plot(dropouts, err_train, 'ob', label='train')
# plt.legend()
# plt.show()
