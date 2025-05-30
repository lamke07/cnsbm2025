import warnings
import numpy as np
from collections import Counter

import jax.numpy as jnp
from jax.scipy.special import entr
from sklearn.cluster import SpectralClustering, SpectralBiclustering

def row_col_redundant(phi_g, phi_h, dir_mean, empty_threshold=1e-6, verbose=True, posterior_map=False):
    # check empty row or column clusters: find clusters that never get any assignment probability
    g_empty = jnp.argwhere(phi_g.max(axis=0) < empty_threshold)
    h_empty = jnp.argwhere(phi_h.max(axis=0) < empty_threshold)
    # go by MAP only (would never get assigned if MAP is used)
    if posterior_map:
        g_labels_unique = jnp.unique(phi_g.argmax(axis=1)).tolist()
        h_labels_unique = jnp.unique(phi_h.argmax(axis=1)).tolist()
        g_empty = jnp.expand_dims(jnp.asarray([i for i in range(phi_g.shape[1]) if i not in g_labels_unique]), 1)
        h_empty = jnp.expand_dims(jnp.asarray([i for i in range(phi_h.shape[1]) if i not in h_labels_unique]), 1)

    # check number of unique clusters
    cluster_means = np.array(jnp.argmax(dir_mean, axis=2))
    _, g_unique_clusters = jnp.unique(cluster_means, axis=0, return_index=True)
    _, h_unique_clusters = jnp.unique(cluster_means, axis=1, return_index=True)
    g_duplicated = [i for i in range(phi_g.shape[1]) if i not in g_unique_clusters]
    h_duplicated = [i for i in range(phi_h.shape[1]) if i not in h_unique_clusters]

    if verbose:
        print(f"Empty rows: {len(g_empty)} | Empty cols: {len(h_empty)} / Duplicated rows: {len(g_duplicated)} | Duplicated cols: {len(h_duplicated)}")

    return g_empty, g_duplicated, h_empty, h_duplicated

def select_redundant_clusters(g_empty, g_duplicated, h_empty, h_duplicated, priority=None, verbose=True):
    # If no proposals available
    if g_empty.shape[0] == 0 and h_empty.shape[0] == 0 and len(g_duplicated) == 0 and len(h_duplicated) == 0:
        print("No empty or duplicated clusters. No more proposals available. Consider creating new clusters")
        return None, None

    # For single splitting (one of row/col), select based on row/col priority or available redundant clusters
    if h_empty.shape[0] == 0 and len(h_duplicated) == 0:
        split_direction = 'row'
        split_cond = 'empty' if g_empty.shape[0] != 0 else 'duplicated'
    elif g_empty.shape[0] == 0 and len(g_duplicated) == 0:
        split_direction = 'col'
        split_cond = 'empty' if h_empty.shape[0] != 0 else 'duplicated'
    elif priority == 'row' or priority is None:
        split_direction = 'row'
        split_cond = 'empty' if g_empty.shape[0] != 0 else 'duplicated'
    elif priority == 'col':
        split_direction = 'col'
        split_cond = 'empty' if h_empty.shape[0] != 0 else 'duplicated'

    cluster_select_g = g_empty.squeeze(1).tolist() if split_cond == 'empty' else g_duplicated
    cluster_select_h = h_empty.squeeze(1).tolist() if split_cond == 'empty' else h_duplicated
    redundant_clusters = [cluster_select_g, cluster_select_h]

    if verbose:
        # print(f"Duplicated rows: {len(g_duplicated)} | Duplicated cols: {len(h_duplicated)}")
        print("Single plitting style: " + split_cond + " " + split_direction)
        print("Available rows/cols to fill: rows", cluster_select_g, "| cols", cluster_select_h)

    return split_direction, redundant_clusters

def clusters_to_split_single(dir_mean, split_direction, empty_threshold=1e-6, min_split_threshold=1e-6, verbose=True):
    # normalized entropy: mask out full entropy clusters (uniform distributions)
    num_cat = dir_mean.shape[2]
    dir_mean_entr = entr(dir_mean).sum(axis=2)/np.log(num_cat)
    dir_mean_entr_masked = jnp.where(dir_mean_entr < 1-empty_threshold, dir_mean_entr, 0)

    # mean and average normalized entropy
    row_entr_avg, row_entr_var = dir_mean_entr_masked.mean(axis=1), dir_mean_entr.var(axis=1)
    col_entr_avg, col_entr_var = dir_mean_entr_masked.mean(axis=0), dir_mean_entr.var(axis=0)

    # choose row and columns with highest entropy
    row_entr_masked = jnp.where(row_entr_avg < 1-empty_threshold, row_entr_avg, 0)
    col_entr_masked = jnp.where(col_entr_avg < 1-empty_threshold, col_entr_avg, 0)

    if split_direction == "row":
        # select entry for which the row is going to be split based on
        tmp = jnp.argsort(row_entr_masked, descending=True)
        row_candidate = tmp[row_entr_masked[tmp] > min_split_threshold]
        col_candidate = jnp.argmax(dir_mean_entr_masked[row_candidate, :], 1)
    else:
        tmp = jnp.argsort(col_entr_masked, descending=True)
        col_candidate = tmp[col_entr_masked[tmp] > min_split_threshold]
        row_candidate = jnp.argmax(dir_mean_entr_masked[:, col_candidate], 0)

    if verbose:
      print(f"Cluster to split up: {list(zip(row_candidate.tolist(), col_candidate.tolist()))}")
      # print("Cluster Probabilities", np.round(dir_mean[row_candidate, col_candidate], 3))

    return list(zip(row_candidate, col_candidate))

def new_cluster_labels_single(
        C, dir_mean, g_labels, h_labels,
        cluster_candidates, split_direction, redundant_clusters,
        how='spectral', split_threshold=None, verbose=True, multi_split=False
    ):
    # get max number of new clusters to select
    cluster_replace = redundant_clusters[0] if split_direction == 'row' else redundant_clusters[1]
    n_iters = min(len(cluster_replace), len(cluster_candidates)) if multi_split else 1
        
    g_labels_new, h_labels_new = g_labels, h_labels
    for i in range(n_iters):
        row_select, col_select = cluster_candidates[i]

        # indices of data submatrix
        idx = jnp.argwhere(g_labels == row_select).squeeze(1)
        idy = jnp.argwhere(h_labels == col_select).squeeze(1)
        sub_C = C[jnp.ix_(idx, idy)] if (split_direction == 'row' or how == 'spectral_bi') else C[jnp.ix_(idx, idy)].transpose()
        if verbose:
            print(split_direction + " to be replaced: " + str(cluster_replace[i]) + f" | using cluster {(int(row_select), int(col_select))}")
            print("Size of data subset:", sub_C.shape)
        
        if len(jnp.unique(sub_C)) <= 1:
            print("Unique cluster, moving on to next")
            continue

        # obtain new splitting labels
        try:
            if how == 'spectral':
                sc_new = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=None).fit(sub_C).labels_
                new_cluster_ids = (sc_new == 1)
            elif how == 'max_split':
                assert split_threshold is not None
                new_cat = jnp.argmax(dir_mean[row_select, col_select])
                new_cluster_ids = (sub_C == new_cat).mean(1) > split_threshold
                if verbose:
                    print(f"New category to be used: {new_cat}")
            elif how == 'random':
                new_cluster_ids = np.random.randint(2, size=sub_C.shape[0])
        except Exception as e:
            warnings.warn(f"Iteration {i}: Error in cluster assignment, skipping to next...")
            continue

        # update cluster labels
        if split_direction == 'row':
            g_labels_new = g_labels_new.at[idx].set(jnp.where(new_cluster_ids, cluster_replace[i], row_select))
        else:
            h_labels_new = h_labels_new.at[idy].set(jnp.where(new_cluster_ids, cluster_replace[i], col_select))

    return g_labels_new, h_labels_new

def get_cluster_sizes(g_labels, h_labels, K, L):
    cluster_sizes = np.zeros((K, L), dtype=int)

    # Iterate through each combination of row and column clusters
    for k in range(K):
        for l in range(L):
            # Count the number of (row, column) pairs belonging to cluster (k, l)
            cluster_sizes[k, l] = jnp.sum((g_labels[:, None] == k) & (h_labels[None, :] == l))
    
    # Flatten the cluster_sizes array to get (k, l) indices and their corresponding sizes
    flattened_indices = [(k, l) for k in range(K) for l in range(L)]
    flattened_sizes = cluster_sizes.flatten()
    
    # Sort the clusters by size in descending order
    sorted_indices = [flattened_indices[i] for i in jnp.argsort(flattened_sizes)[::-1]]
    sorted_sizes = sorted(flattened_sizes, reverse=True)

    return cluster_sizes, sorted_indices, sorted_sizes

def new_cluster_labels_block(
    C, g_labels, h_labels,
    cluster_candidates, redundant_clusters,
    how='spectral_bi', verbose=True, multi_split=False
    ):
    # get max number of new clusters to select
    n_iters = min(len(redundant_clusters[0]), len(redundant_clusters[1]), len(cluster_candidates)) if multi_split else 1
        
    g_labels_new = g_labels
    h_labels_new = h_labels

    for i in range(n_iters):
        row_select, col_select = cluster_candidates[i]

        if verbose:
            print(f"Cluster to be replaced: {redundant_clusters[0][i]} {redundant_clusters[1][i]} | using cluster {(int(row_select), int(col_select))}")

        # indices of data submatrix
        idx = jnp.argwhere(g_labels == row_select).squeeze(1)
        idy = jnp.argwhere(h_labels == col_select).squeeze(1)
        sub_C = C[jnp.ix_(idx, idy)] if (how == 'spectral_bi') else C[jnp.ix_(idx, idy)].transpose()
        print("Sub data size:", sub_C.shape)

        if len(jnp.unique(sub_C)) <= 1:
            print("Unique cluster, moving on to next")
            continue

        # obtain new splitting labels
        try:
            model = SpectralBiclustering(n_clusters=(2,2), method="log", random_state=0)
            model.fit(sub_C)
            new_cluster_ids_g = (model.row_labels_ == 1)
            new_cluster_ids_h = (model.column_labels_ == 1)
            # add random or maxsplit?
        except Exception as e:
            warnings.warn(f"Iteration {i}: Error in cluster assignment, skipping to next...")
            continue

        if how == 'spectral_bi':
            g_labels_new = g_labels_new.at[idx].set(jnp.where(new_cluster_ids_g, redundant_clusters[0][i], row_select))
            h_labels_new = h_labels_new.at[idy].set(jnp.where(new_cluster_ids_h, redundant_clusters[1][i], col_select))

    return g_labels_new, h_labels_new

def row_col_impurity(dir_mean, empty_threshold=1e-6, entr_threshold=0.2, verbose=True):
    # normalized entropy: mask out full entropy clusters (uniform distributions)
    num_cat = dir_mean.shape[2]
    dir_mean_entr = entr(dir_mean).sum(axis=2)/np.log(num_cat)
    dir_mean_entr_masked = jnp.where(dir_mean_entr < 1-empty_threshold, dir_mean_entr, 0)

    # average and variance normalized entropy
    row_entr_avg, row_entr_var = dir_mean_entr.mean(axis=1), dir_mean_entr.var(axis=1)
    col_entr_avg, col_entr_var = dir_mean_entr.mean(axis=0), dir_mean_entr.var(axis=0)

    # choose row and columns with highest entropy
    row_entr_masked = jnp.where(row_entr_avg < 1-empty_threshold, row_entr_avg, 0)
    col_entr_masked = jnp.where(col_entr_avg < 1-empty_threshold, col_entr_avg, 0)

    # return "impurities" based on a given threshold
    row_impurities = jnp.argwhere(row_entr_masked > entr_threshold).squeeze(1).tolist()
    col_impurities = jnp.argwhere(col_entr_masked > entr_threshold).squeeze(1).tolist()

    if verbose:
      print(f"Mean row entropy: {row_entr_avg[row_entr_avg < 1 - empty_threshold].mean():,.3f} | Mean col entropy: {col_entr_avg[col_entr_avg < 1 - empty_threshold].mean():,.3f}")
      print(f"Cluster Impurities: Row {row_impurities}, Col {col_impurities}\n")

    return row_impurities, col_impurities, dir_mean_entr_masked

def reset_min_clusters(g_labels, h_labels, gamma_kl, alpha_pi, min_samples_g=15, min_samples_h=15, verbose=True, direction='multi'):
    g_counter, h_counter = Counter(g_labels), Counter(h_labels)
    
    g_min = [k for k, v in g_counter.items() if v < min_samples_g]
    h_min = [k for k, v in h_counter.items() if v < min_samples_h]

    if direction == 'multi':
        pass
    elif len(h_min) == 0 and len(g_min) > 0:
        g_min = [g_min[np.random.choice(range(len(g_min)))]]
        h_min = []
    elif len(g_min) == 0 and len(h_min) > 0:
        g_min = []
        h_min = [h_min[np.random.choice(range(len(h_min)))]]
    elif len(g_min) > 0 and len(h_min) > 0:
        if direction == 'row':
            g_min = [g_min[np.random.choice(range(len(g_min)))]]
            h_min = []
        else:
            g_min = []
            h_min = [h_min[np.random.choice(range(len(h_min)))]]

    if verbose:
        print(f"Resetting small sample row clusters {g_min}, col clusters {h_min}")
        for k in g_min:
            print(f"Row cluster {k}: {g_counter[k]} samples")
        for k in h_min:
            print(f"Col cluster {k}: {h_counter[k]} samples")

    for x in g_min:
        gamma_kl = gamma_kl.at[x, :, :].set(alpha_pi)
    
    for y in h_min:
        gamma_kl = gamma_kl.at[:, y, :].set(alpha_pi)

    return gamma_kl

def reset_duplicated_clusters(phi_g, phi_h, dir_mean, gamma_kl, alpha_pi, empty_threshold=1e-6, verbose=True, posterior_map=True, direction='multi'):
    g_empty, g_duplicated, h_empty, h_duplicated = row_col_redundant(phi_g, phi_h, dir_mean, empty_threshold=empty_threshold, verbose=verbose, posterior_map=posterior_map)

    g_reset = list(set(g_duplicated) - set(g_empty.squeeze(1).tolist()))
    h_reset = list(set(h_duplicated) - set(h_empty.squeeze(1).tolist()))

    if direction == 'multi':
        pass
    elif len(h_reset) == 0 and len(g_reset) > 0:
        g_reset = [g_reset[np.random.choice(range(len(g_reset)))]]
        h_reset = []
    elif len(g_reset) == 0 and len(h_reset) > 0:
        g_reset = []
        h_reset = [h_reset[np.random.choice(range(len(h_reset)))]]
    elif len(g_reset) > 0 and len(h_reset) > 0:
        if direction == 'row':
            g_reset = [g_reset[np.random.choice(range(len(g_reset)))]]
            h_reset = []
        else:
            g_reset = []
            h_reset = [h_reset[np.random.choice(range(len(h_reset)))]]

    if verbose:
        print(f"Resetting duplicated row clusters {g_reset}, col clusters {h_reset}")

    for x in g_reset:
        gamma_kl = gamma_kl.at[x, :, :].set(alpha_pi)
    
    for y in h_reset:
        gamma_kl = gamma_kl.at[:, y, :].set(alpha_pi)

    return gamma_kl
