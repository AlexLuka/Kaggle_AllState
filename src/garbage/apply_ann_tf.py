import src.data_reader
import src.data_export
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


"""
The first attempt to create an ANN model with TF.

"""


# Create model
def multilayer_net(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


def execute_nn_4l(x_train, x_test, y_train, y_test,
                  learning_rate=0.1,
                  training_epochs=20,
                  batch_size=100,
                  layer_size=(40, 30, 20, 10),
                  is_test=False,
                  test_number=0):

    # NN Parameters
    # learning_rate = 0.1
    # training_epochs = 20
    # batch_size = 100
    display_step = 10

    # Network Parameters
    n_hidden_1 = layer_size[0]
    n_hidden_2 = layer_size[1]
    n_hidden_3 = layer_size[2]
    n_hidden_4 = layer_size[3]
    n_input = x_train.shape[1]
    total_len = x_train.shape[0]
    n_classes = 1

    print 'Number of examples: {}'.format(total_len)
    print 'Number of features: {}'.format(n_input)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
        'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
        'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
    }

    # Construct model
    model = multilayer_net(x, weights, biases)

    # Define loss and optimizer. Tricky part, model has to be transposed
    cost = tf.reduce_mean(tf.abs(tf.sub(tf.transpose(model), y)))

    # Tried two different optimizers
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(total_len/batch_size)
            # Loop over all batches
            batch_y = []
            estimate = []
            for i in range(total_batch-1):
                batch_x = x_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train[i*batch_size:(i+1)*batch_size]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, estimate = sess.run([optimizer, cost, model], feed_dict={x: batch_x, y: batch_y})

                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if epoch % display_step == 0:
                print 'Epoch: {:04d}\tcost={:.9f}'.format(epoch, avg_cost)

        print ("Optimization Finished!")
        print '[*]' + '=' * 40

        # Train accuracy
        _, mae_train, predicted_train = sess.run([optimizer, cost, model], feed_dict={x: x_train, y: y_train})
        print "Train mean abs error = {}. Predicted (MIN,MAX)=({},{})".format(mae_train,
                                                                              min(predicted_train),
                                                                              max(predicted_train))

        # CV accuracy
        _, mae_cv, predicted_cv = sess.run([optimizer, cost, model], feed_dict={x: x_test, y: y_test})
        print "Test mean abs error = {}. Predicted (MIN,MAX)=({},{})".format(mae_cv,
                                                                             min(predicted_cv),
                                                                             max(predicted_cv))

        if is_test:
            # Test
            test_id = src.data_reader.get_test_id()
            test_num = src.data_reader.get_test_num()
            test_cat = src.data_reader.get_test_cat()

            TestArr = np.hstack((test_num, test_cat))
            TestLab = [0]*TestArr.shape[0]

            _, _, predicted = sess.run([optimizer, cost, model], feed_dict={x: TestArr, y: TestLab})
            predicted = np.exp(predicted)
            src.data_export.save_submission_ann(test_id, predicted, test_number)
        return mae_train, mae_cv


def main_f():
    m_tr = []
    m_cv = []

    # Get the data and split it on train and cv sets
    num_train = src.data_reader.get_train_num()
    cat_train = src.data_reader.get_train_cat()
    loss_train = np.log(src.data_reader.get_train_loss())

    x_tr, x_tst, y_tr, y_tst = train_test_split(np.hstack((num_train, cat_train)),
                                                loss_train,
                                                test_size=0.3,
                                                random_state=np.random.randint(np.iinfo(np.int32).max))

    for bs in [1000]:           # [100, 200, 500, 1000, 2000, 5000, 10000]:
        m1, m2 = execute_nn_4l(x_tr, x_tst, y_tr, y_tst,
                               learning_rate=0.0001,
                               training_epochs=2000,
                               batch_size=bs,
                               layer_size=(140, 100, 50, 10),
                               is_test=True,
                               test_number=12)
        m_tr.append(m1)
        m_cv.append(m2)

    plt.plot(m_tr, 'ok', label='train')
    plt.plot(m_cv, 'or', label='cv')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main_f()
