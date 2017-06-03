from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import data_reader
import numpy as np
from time import time


shift = 200

train_id, train_data, train_loss = data_reader.train_data_reader2()

tr_loss = np.log(train_loss + shift)
mu = np.mean(tr_loss)
sd = np.std(tr_loss)
tr_loss = (tr_loss - mu) / sd

X_train, X_test, y_train, y_test = train_test_split(train_data,
                                                    tr_loss,
                                                    test_size=0.3,
                                                    random_state=np.random.randint(np.iinfo(np.int32).max))

print 'Data loaded and split'
#   SVM Regression
print 'Train regression model with SVM'
t0 = time()
svr = SVR(kernel='poly', degree=3, verbose=True)
svr.fit(X_train, y_train)
print 'Training takes {} minutes'.format((time() - t0)/60.0)
print 'Predict result'
y_predicted = svr.predict(X_test)

print 'Error: {}'.format(mean_absolute_error(np.exp(sd*y_test+mu) - shift, np.exp(sd*y_predicted+mu) - shift))
