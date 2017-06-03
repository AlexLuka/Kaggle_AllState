import numpy as np
import data_reader
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt


"""
 If I remember correctly I tried to detect outliers with
 Replicator Neural Network (RNN). Found the idea in this
 study: https://togaware.com/papers/tr02102.pdf
 It actually detects some outliers, however these outliers
 are not in tails of the target distribution but within it.
 So, probably it is able to detect 'unusual' examples


"""


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
        y_batch = y[batch_index, :].toarray()

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
             n1, n2,
             dropout1=0.4,
             dropout2=0.4):
    model = Sequential()

    model.add(Dense(n1, input_dim=input_size, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout1))

    model.add(Dense(n2, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout2))

    model.add(Dense(n1, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout1))

    model.add(Dense(input_size, init='he_normal', activation='linear'))

    model.compile(loss='mse', optimizer='adadelta')
    return model


def main_f():
    # ====================== Actual run ===================================
    # performing tuning here
    # important params:
    shift_in = 220          # possible values are 200,220
    epochs = 50
    run = 100

    # train data
    print 'Loading training data'
    train_id, train_data, train_loss = data_reader.train_data_reader2()

    # mark outliers as 1
    loss_outliers = np.zeros(train_loss.shape)
    loss_outliers[np.where(train_loss > 40000)] = 1
    loss_outliers = np.asarray(loss_outliers, dtype=bool)

    print 'Loading test data'
    test_id, test_data = data_reader.test_data_reader2()

    predict_res = np.zeros(shape=(test_id.shape[0], 1), dtype=float)

    m = nn_model(train_data.shape[1],
                 200, 50,
                 dropout1=0.4,
                 dropout2=0.2)

    m.fit_generator(generator=batch_generator(train_data, train_data, 256, True),
                    nb_epoch=epochs,
                    samples_per_epoch=train_data.shape[0],
                    verbose=1
                    )

    train_predict = m.predict_generator(generator=batch_generator_predict(train_data, 1000),
                                        val_samples=train_data.shape[0])

    corr_coeff = []

    for val1, val2 in zip(train_predict, train_data):
        val2 = val2.toarray()

        corr_coeff.append(np.corrcoef(val1, val2)[0, 1])

    del train_data, train_predict

    print 'Done here'
    corr_coeff = np.asarray(corr_coeff)

    print 'Plot loss'
    plt.plot(corr_coeff, train_loss, '.')
    plt.plot(corr_coeff[loss_outliers], train_loss[loss_outliers], '.r')
    plt.show()


if __name__ == '__main__':
    main_f()
