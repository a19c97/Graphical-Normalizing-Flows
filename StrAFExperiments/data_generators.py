import numpy as np


def generate_adj_mat(dim, threshold):
    """Generates an adjacency matrix."""
    A = np.random.uniform(size=(dim, dim))
    A = (A > threshold).astype(int)
    A = np.tril(A, -1)
    return A


def generate_weight_mat(adj_mat, w_range):
    """Generates weight matrix describing relationships between nodes.
    
    Args:
        adj_mat (np.array): D x D matrix describing adjacencies.
        w_range (int, int): Min and max value of weights.
    """
    weights = np.random.uniform(w_range[0], w_range[1], size=adj_mat.shape)
    weights = adj_mat * weights
    return weights


def generate_multimodal_dataset(w_mat, n_samp, n_modes, mean_range, std_range):
    """Generates dataset with multimodal marginal distributions.
    
    Args:
        w_mat (np.array): D x D matrix describing relations between variables.
        n_samp (int): Number of samples to generate.
        n_modes (int): Number of modes in each marginal distribution.
    
    Returns:
        (np.array) D x N matrix of sampled data.
    """
    data_arr = np.zeros((w_mat.shape[0], n_samp))

    # Tracks data distribution
    mode_mixture_weights = {}
    mode_means = {}
    mode_stds = {}

    for i, row in enumerate(w_mat):
        # Check if variable has dependencies
        weights = w_mat[i][w_mat[i] != 0]
        if len(weights) == 0:
            # Generate independent variable
            mixture_weights = np.random.dirichlet(alpha=np.ones(n_modes))
            means = np.random.uniform(*mean_range, n_modes)
            stds = np.random.uniform(*std_range, n_modes)

            mode_means[i] = means
            mode_stds[i] = stds
            mode_mixture_weights[i] = mixture_weights

            mode_data = []
            for m in range(n_modes):
                # Adds additional sample as a fix for rounding
                mode_samp = int(n_samp * mixture_weights[m]) + 1
                mode = np.random.normal(means[m], stds[m], mode_samp)
                mode_data.append(mode)
            row_data = np.concatenate(mode_data)[:n_samp]
            data_arr[i] = row_data
        else:
            row_mask = row != 0
            prior_rows_data = data_arr[row_mask]

            nonzero_weights = row[row != 0]

            nz_weights = nonzero_weights[:, np.newaxis]
            nz_weights = np.repeat(nz_weights, n_samp, axis=-1)

            new_row_data = np.multiply(nz_weights, prior_rows_data)

            new_row_data = new_row_data ** 2
            new_row_data = np.sum(new_row_data, axis=0)
            new_row_data = new_row_data ** 0.5
            
            new_row_data += np.random.normal(size=n_samp)

            data_arr[i] = new_row_data

    dist_args = {"means": mode_means,
                 "stds": mode_stds,
                 "mixture_weights": mode_mixture_weights}

    return data_arr, dist_args


def normalize_data(data):
    norm_data = np.zeros((data.shape[0], data.shape[1]))
    for i, row in enumerate(data):
        norm_data[i] = (row - row.mean()) / row.std()
    return norm_data
