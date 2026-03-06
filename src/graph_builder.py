import numpy as np
import scipy.sparse as sp
import torch

def build_graph(interactions, num_users, num_items):

    users = [x[0] for x in interactions]
    items = [x[1] for x in interactions]

    data = np.ones(len(users))

    R = sp.coo_matrix(
        (data, (users, items)),
        shape=(num_users, num_items)
    )

    upper = sp.hstack([sp.csr_matrix((num_users, num_users)), R])
    lower = sp.hstack([R.T, sp.csr_matrix((num_items, num_items))])
    A = sp.vstack([upper, lower]).tocsr()

    # Normalize
    rowsum = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv = sp.diags(d_inv_sqrt)

    A_norm = D_inv.dot(A).dot(D_inv).tocoo()

    indices = torch.LongTensor([A_norm.row, A_norm.col])
    values = torch.FloatTensor(A_norm.data)
    shape = torch.Size(A_norm.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

