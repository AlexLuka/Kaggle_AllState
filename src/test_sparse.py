from scipy.sparse import csr_matrix, hstack, vstack
import numpy as np
import data_reader
import matplotlib.pyplot as plt


def delete_row_csr(mat, i):
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])


def delete_column_csr(mat, i):
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    print 'Value: ', n
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])


def delete_column_csr1(mat, i):
    print mat[:, :i].shape
    print mat[:, i:].shape
    return hstack((mat[:, :i], mat[:, i+1:]), format='csr')


# test_id, test_data = data_reader.test_data_reader()
# train_id, train_data, train_loss = data_reader.train_data_reader()
#
# cols_to_delete = []
#
# threshold = 1
#
# plt.hist(test_data[:, 1184].toarray(), 50)
# plt.show()

data_reader.data_converter2()

# for col in range(test_data.shape[1]):
#     print 'Column: ', col
#     if reduce(lambda x, y: x+y, test_data[:, col].toarray())[0] < threshold:
#         cols_to_delete.append(col)
#
#     if reduce(lambda x, y: x + y, train_data[:, col].toarray())[0] < threshold:
#         cols_to_delete.append(col)
#
# print set(cols_to_delete)
