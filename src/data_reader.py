# import csv   <- this is fucking inefficient shit
import numpy as np
import cPickle
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.sparse import csr_matrix, hstack


"""
 Combination of public data with my own thoughts.

"""


# AHAHA, later I discovered Pandas for myself)))
def get_train_data():
    file_name = '../data/train.csv'

    with open(file_name, 'rb') as f:
        the_data = dict()

        # get the header
        header = f.readline().split(',')
        for field in header:
            field = field.replace('\n', '')
            the_data[field] = list()
            print field

        # get features
        for row in f.readlines():
            features = row.split(',')
            for field, feature in zip(header, features):
                if 'cat' in field:
                    the_data[field].append(feature)
                elif 'cont' in field:
                    the_data[field].append(float(feature))
                elif 'loss' in field:
                    the_data['loss'].append(float(feature))
                elif 'id' in field:
                    the_data[field].append(int(feature))
    return the_data


def get_test_data():
    file_name = '../data/test.csv'

    with open(file_name, 'rb') as f:
        the_data = dict()

        # get the header
        header = f.readline()
        header = header.replace('\n', '').split(',')
        for field in header:
            the_data[field] = list()

        # get features
        for row in f.readlines():
            features = row.split(',')
            for field, feature in zip(header, features):
                if 'cat' in field:
                    the_data[field].append(feature)
                elif 'cont' in field:
                    the_data[field].append(float(feature))
                elif 'id' in field:
                    the_data[field].append(int(feature))
    return the_data


def convert(data):
    # continuous data
    cont_feats = []
    for key in data.keys():
        if 'cont' in key:
            cont_feats.append(data[key])
    num_features = np.vstack(cont_feats).T

    cat_feats = []
    for key in data.keys():
        if 'cat' in key:
            unique_values = set(data[key])
            lab_encode = LabelEncoder()
            lab_encode.fit(list(unique_values))
            res = lab_encode.transform(data[key])
            cat_feats.append((res - min(res)) / (max(res) - min(res)))
    cat_features = np.vstack(cat_feats).T

    return num_features, cat_features


def get_train_num():
    return cPickle.load(open('../data/train-num-features.pkl', 'rb'))


def get_train_cat():
    return cPickle.load(open('../data/train-cat-features.pkl', 'rb'))


def get_train_loss():
    return cPickle.load(open('../data/train-loss-features.pkl', 'rb'))


def get_test_id():
    return cPickle.load(open('../data/test-id-features.pkl', 'rb'))


def get_test_num():
    return cPickle.load(open('../data/test-num-features.pkl', 'rb'))


def get_test_cat():
    return cPickle.load(open('../data/test-cat-features.pkl', 'rb'))


# ==================================== PANDAS READER ===============================================


def data_converter():
    # read data
    data_train = pd.read_csv('../data/train.csv')
    data_test = pd.read_csv('../data/test.csv')

    # keep ids and losses
    loss = data_train['loss'].values
    id_train = data_train['id'].values
    id_test = data_test['id'].values

    #
    number_train_examples = data_train.shape[0]
    train_test_stack = pd.DataFrame(pd.concat((data_train, data_test), axis=0))

    sparse_data = []
    for col in train_test_stack.columns:
        if 'cat' in col:
            # create dummy variables from categorical data
            dummy = pd.get_dummies(train_test_stack[col].astype('category'))
            # create sparse matrix from dummy vars
            tmp = csr_matrix(dummy)
            # add data do the total data list
            sparse_data.append(tmp)

    numerical_features = [col for col in train_test_stack.columns if 'cont' in col]
    sc = StandardScaler()
    tmp = csr_matrix(sc.fit_transform(train_test_stack[numerical_features]))
    sparse_data.append(tmp)

    del (train_test_stack, data_train, data_test)

    # sparse train and test data
    xtr_te = hstack(sparse_data, format='csr')
    x_train = xtr_te[:number_train_examples, :]
    x_test = xtr_te[number_train_examples:, :]

    print('Dim train', x_train.shape)
    print('Dim test', x_test.shape)

    del (xtr_te, sparse_data, tmp)

    # export data
    df1 = pd.DataFrame()
    df1['id'] = id_train
    df1['loss'] = loss
    df1['data'] = x_train
    df1.to_pickle('../data2/train-data')

    # cPickle.dump(x_train, open('../data2/train-data-cp'))

    df2 = pd.DataFrame()
    df2['id'] = id_test
    df2['data'] = x_test
    df2.to_pickle('../data2/test-data')


# modification 2: remove those categorical features that a present in test set
# but do not present in train set and vise versa
def data_converter2():
    # read data
    data_train = pd.read_csv('../data/train.csv')
    data_test = pd.read_csv('../data/test.csv')

    # keep ids and losses
    loss = data_train['loss'].values
    id_train = data_train['id'].values
    id_test = data_test['id'].values

    #
    number_train_examples = data_train.shape[0]
    train_test_stack = pd.DataFrame(pd.concat((data_train, data_test), axis=0))

    sparse_data = []
    for col in train_test_stack.columns:
        if 'cat' in col:
            # create dummy variables from categorical data
            dummy = pd.get_dummies(train_test_stack[col].astype('category')).as_matrix()

            mask = np.ones(dummy.shape[1], dtype=bool)

            cols_to_delete = []

            for i in range(dummy.shape[1]):
                a = reduce(lambda x, y: x + y, dummy[:number_train_examples, i])    # train set
                b = reduce(lambda x, y: x + y, dummy[number_train_examples:, i])    # test set

                if a == 0 or b == 0:
                    cols_to_delete.append(i)

            if len(cols_to_delete) > 0:
                mask[cols_to_delete] = False
                dummy = dummy[:, mask]

            # create sparse matrix from dummy vars
            tmp = csr_matrix(dummy)

            # add data do the total data list
            sparse_data.append(tmp)

    numerical_features = [col for col in train_test_stack.columns if 'cont' in col]
    sc = StandardScaler()
    tmp = csr_matrix(sc.fit_transform(train_test_stack[numerical_features]))
    sparse_data.append(tmp)

    del (train_test_stack, data_train, data_test)

    # sparse train and test data
    xtr_te = hstack(sparse_data, format='csr')
    x_train = xtr_te[:number_train_examples, :]
    x_test = xtr_te[number_train_examples:, :]

    print('Dim train', x_train.shape)
    print('Dim test', x_test.shape)

    del (xtr_te, sparse_data, tmp)

    # export data
    df1 = pd.DataFrame()
    df1['id'] = id_train
    df1['loss'] = loss
    df1['data'] = x_train
    df1.to_pickle('../data4/train-data')

    df2 = pd.DataFrame()
    df2['id'] = id_test
    df2['data'] = x_test
    df2.to_pickle('../data4/test-data')


# modification 3: add features combinations
# also remove those categorical features that a present in test set
# but do not present in train set and vise versa
def data_converter3():
    features_to_combine = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
                          'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
                          'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
                          'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')

    # read data
    data_train = pd.read_csv('../data/train.csv')
    data_test = pd.read_csv('../data/test.csv')

    # keep ids and losses
    loss = data_train['loss'].values
    id_train = data_train['id'].values
    id_test = data_test['id'].values

    #
    number_train_examples = data_train.shape[0]
    train_test_stack = pd.DataFrame(pd.concat((data_train, data_test), axis=0))

    sparse_data = []
    print 'Convert original features'
    for col in train_test_stack.columns:
        if 'cat' in col:
            # create dummy variables from categorical data
            dummy = pd.get_dummies(train_test_stack[col].astype('category')).as_matrix()

            mask = np.ones(dummy.shape[1], dtype=bool)

            cols_to_delete = []

            for i in range(dummy.shape[1]):
                a = reduce(lambda x, y: x + y, dummy[:number_train_examples, i])    # train set
                b = reduce(lambda x, y: x + y, dummy[number_train_examples:, i])    # test set

                if a == 0 or b == 0:
                    cols_to_delete.append(i)

            if len(cols_to_delete) > 0:
                mask[cols_to_delete] = False
                dummy = dummy[:, mask]

            # create sparse matrix from dummy vars
            tmp = csr_matrix(dummy)

            # add data do the total data list
            sparse_data.append(tmp)

    print 'Feature combinations'
    for comb in itertools.combinations(features_to_combine, 2):
        feat = comb[0] + "_" + comb[1]
        print '\tCombination: {}'.format(feat)

        # create dummy variables from categorical data
        dummy = pd.get_dummies((train_test_stack[comb[0]] +
                                train_test_stack[comb[1]]).astype('category')).as_matrix()

        mask = np.ones(dummy.shape[1], dtype=bool)
        cols_to_delete = []

        for i in range(dummy.shape[1]):
            a = reduce(lambda x, y: x + y, dummy[:number_train_examples, i])  # train set
            b = reduce(lambda x, y: x + y, dummy[number_train_examples:, i])  # test set

            if a == 0 or b == 0:
                cols_to_delete.append(i)

        if len(cols_to_delete) > 0:
            mask[cols_to_delete] = False
            dummy = dummy[:, mask]

        # create sparse matrix from dummy vars
        tmp = csr_matrix(dummy)

        # add data do the total data list
        sparse_data.append(tmp)

    print 'Scale numerical features'
    numerical_features = [col for col in train_test_stack.columns if 'cont' in col]
    sc = StandardScaler()
    tmp = csr_matrix(sc.fit_transform(train_test_stack[numerical_features]))
    sparse_data.append(tmp)

    del (train_test_stack, data_train, data_test)

    print 'Stack sparse data'
    # sparse train and test data
    xtr_te = hstack(sparse_data, format='csr')
    x_train = xtr_te[:number_train_examples, :]
    x_test = xtr_te[number_train_examples:, :]

    print('Dim train', x_train.shape)
    print('Dim test', x_test.shape)

    del (xtr_te, sparse_data, tmp)

    # export data
    df1 = pd.DataFrame()
    df1['id'] = id_train
    df1['loss'] = loss
    df1['data'] = x_train
    df1.to_pickle('../data7/train-data')

    df2 = pd.DataFrame()
    df2['id'] = id_test
    df2['data'] = x_test
    df2.to_pickle('../data7/test-data')


# modification 4: remove those categorical features that a present in test set
# but do not present in train set and vise versa
# combinations of numerical features
def data_converter4():
    # read data
    data_train = pd.read_csv('../data/train.csv')
    data_test = pd.read_csv('../data/test.csv')

    # keep ids and losses
    loss = data_train['loss'].values
    id_train = data_train['id'].values
    id_test = data_test['id'].values

    #
    number_train_examples = data_train.shape[0]
    train_test_stack = pd.DataFrame(pd.concat((data_train, data_test), axis=0))

    sparse_data = []
    for col in train_test_stack.columns:
        if 'cat' in col:
            # create dummy variables from categorical data
            dummy = pd.get_dummies(train_test_stack[col].astype('category')).as_matrix()

            mask = np.ones(dummy.shape[1], dtype=bool)

            cols_to_delete = []

            for i in range(dummy.shape[1]):
                a = reduce(lambda x, y: x + y, dummy[:number_train_examples, i])    # train set
                b = reduce(lambda x, y: x + y, dummy[number_train_examples:, i])    # test set

                if a == 0 or b == 0:
                    cols_to_delete.append(i)

            if len(cols_to_delete) > 0:
                mask[cols_to_delete] = False
                dummy = dummy[:, mask]

            # create sparse matrix from dummy vars
            tmp = csr_matrix(dummy)

            # add data do the total data list
            sparse_data.append(tmp)

    num_data = []
    numerical_features = [col for col in train_test_stack.columns if 'cont' in col]
    for num_feat in numerical_features:
        num_data.append(train_test_stack[num_feat].values)

    for comb in itertools.combinations(numerical_features, 2):
        feat = comb[0] + '_' + comb[1]
        print '\tCombination {}'.format(feat)
        comb_val = np.multiply(train_test_stack[comb[0]].values, train_test_stack[comb[1]].values)
        num_data.append(comb_val)

    print 'Total number of numerical features: {}'.format(len(num_data))

    print num_data
    print len(num_data[0])
    print len(num_data)

    num_tot = np.vstack(num_data).T

    print 'THE SHAPE = {}'.format(num_tot.shape)

    sc = StandardScaler()
    tmp = csr_matrix(sc.fit_transform(num_tot))
    sparse_data.append(tmp)

    del (train_test_stack, data_train, data_test)

    # sparse train and test data
    xtr_te = hstack(sparse_data, format='csr')
    x_train = xtr_te[:number_train_examples, :]
    x_test = xtr_te[number_train_examples:, :]

    print('Dim train', x_train.shape)
    print('Dim test', x_test.shape)

    del (xtr_te, sparse_data, tmp)

    # export data
    df1 = pd.DataFrame()
    df1['id'] = id_train
    df1['loss'] = loss
    df1['data'] = x_train
    df1.to_pickle('../data7/train-data')

    df2 = pd.DataFrame()
    df2['id'] = id_test
    df2['data'] = x_test
    df2.to_pickle('../data7/test-data')


def test_data_reader():
    ddt = pd.read_pickle('../data2/test-data')
    return ddt['id'].values, ddt['data'].values[0]


def train_data_reader():
    ddt = pd.read_pickle('../data2/train-data')
    return ddt['id'].values, ddt['data'].values[0], ddt['loss'].values


def test_data_reader2():
    ddt = pd.read_pickle('../data4/test-data')
    return ddt['id'].values, ddt['data'].values[0]


def train_data_reader2():
    ddt = pd.read_pickle('../data4/train-data')
    return ddt['id'].values, ddt['data'].values[0], ddt['loss'].values


def test_data_reader3():
    ddt = pd.read_pickle('../data6/test-data')
    return ddt['id'].values, ddt['data'].values[0]


def train_data_reader3():
    ddt = pd.read_pickle('../data6/train-data')
    return ddt['id'].values, ddt['data'].values[0], ddt['loss'].values


def test_data_reader7():
    ddt = pd.read_pickle('../data7/test-data')
    return ddt['id'].values, ddt['data'].values[0]


def train_data_reader7():
    ddt = pd.read_pickle('../data7/train-data')
    return ddt['id'].values, ddt['data'].values[0], ddt['loss'].values
