import torch
import numpy as np


def load_adj_mtx():
    A = torch.load('./adj_mtx/hepmass1.pt').numpy()
    # For hepmass, swap sums[17] and sums[18]
    # A = torch.load('./adj_mtx/power1.pt').numpy()
    # A = torch.load('./adj_mtx/gas1.pt').numpy()
    A = np.where(A > 0, 1, 0)
    row_sums = A.sum(axis=1)
    sums = list(enumerate(row_sums))
    sums.sort(key=lambda x: x[1])
    import pdb; pdb.set_trace()

    # Create permutation matrix
    P = np.zeros(A.shape)
    for idx, (col, _) in enumerate(sums):
        P[col][idx] = 1
    import pdb; pdb.set_trace()

    # Permute adjacency matrix
    A = np.matmul(np.matmul(P.T, A), P)
    import pdb; pdb.set_trace()

    return A

def make_permutation_matrix(dataset_name):
    print(f"Making P for {dataset_name}")
    if dataset_name == 'power':
        P = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0]
        ])
        path_to_P = "./adj_mtx/power_perm_1"
    elif dataset_name == "gas":
        P = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0]
        ])
        path_to_P = "./adj_mtx/gas_perm_1"

    np.savez(path_to_P, P)
    print(f"Saved P to {path_to_P}")

if __name__ == '__main__':
    # make_permutation_matrix("gas")
    load_adj_mtx()