import functools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.scipy.special import digamma, gammaln, entr
from functools import partial

from sklearn.cluster import SpectralClustering, SpectralBiclustering

"""VI Update Functions"""
# Helper functions
def normalize_log_probs_row(log_probs):
    """Normalize log probabilities to sum to 1."""
    max_log = jnp.max(log_probs, axis=-1, keepdims=True)
    stabilized = log_probs - max_log
    return jnp.exp(stabilized) / jnp.sum(jnp.exp(stabilized), axis=-1, keepdims=True)

def normalize_log_probs(log_probs):
    """
    Converts a matrix of log probabilities to normalized probabilities for each row.

    Args:
        log_probs (jnp.ndarray): A matrix of shape (N, K) containing log probabilities.

    Returns:
        jnp.ndarray: A matrix of shape (N, K) containing normalized probabilities for each row.
    """
    max_log_probs = jnp.max(log_probs, axis=1, keepdims=True)
    exp_shifted = jnp.exp(log_probs - max_log_probs)
    row_sums = jnp.sum(exp_shifted, axis=1, keepdims=True)

    normalized_probs = exp_shifted / row_sums

    return normalized_probs

################################################
# Local updates for update_phi_g
@jax.jit
def update_phi_g_full(phi_h, gamma_kl_di, gamma_kl_sum_di, gamma_g_di, c_row):
    """Update q(g_i | phi^g)."""
    # phi_tmp = phi_h.transpose(1,0)[jnp.newaxis,...]
    # tmp1 = gamma_kl_di[:, :, c_row]
    # out = (phi_tmp*(tmp1-gamma_kl_sum_di)).sum(axis=(1,2)) - gamma_g_di
    
    out = jnp.einsum("ml, klm -> k", phi_h, gamma_kl_di[:, :, c_row]-gamma_kl_sum_di, precision='highest') - gamma_g_di

    return out

@jax.jit
def update_phi_g_mis(phi_h, gamma_kl_di, gamma_kl_sum_di, gamma_g_di, c_row, c_row_mask):
    """Update q(g_i | phi^g)."""
    # phi_tmp = phi_h.transpose(1,0)[jnp.newaxis,...]
    # tmp1 = gamma_kl_di[:, :, c_row]
    # inner_sum = (phi_tmp*(tmp1-gamma_kl_sum_di)).sum(axis=1)
    # out = (c_row_mask[jnp.newaxis,:] * inner_sum).sum(axis=1) - gamma_g_di
    # # tmp_mask = (c_row != -1)[jnp.newaxis, jnp.newaxis,:]
    # # out = (phi_tmp*(tmp1-gamma_kl_sum_di)*tmp_mask).sum(axis=(1,2)) - gamma_g_di
    inner_sum = jnp.einsum("ml, klm -> km", phi_h, gamma_kl_di[:, :, c_row]-gamma_kl_sum_di, precision='highest')
    out = (jnp.einsum("m,km -> k", c_row_mask, inner_sum, precision='highest') - gamma_g_di)

    return out

vmap_update_phi_g_full = jax.vmap(update_phi_g_full, in_axes=(None, None, None, None, 0))
vmap_update_phi_g_mis = jax.vmap(update_phi_g_mis, in_axes=(None, None, None, None, 0, 0))
jax.jit(vmap_update_phi_g_full)
jax.jit(vmap_update_phi_g_mis)

# Local updates for update_phi_h
@jax.jit
def update_phi_h_full(phi_g, gamma_kl_di, gamma_kl_sum_di, gamma_h_di, c_col):
    """q(h_j | phi^h)."""
    # phi_tmp = phi_g.transpose(1,0)[:, jnp.newaxis, :]
    # tmp1 = gamma_kl_di[:, :, c_col]
    # out = (phi_tmp*(tmp1-gamma_kl_sum_di)).sum(axis=(0,2)) - gamma_h_di

    out = jnp.einsum("nk, kln -> l", phi_g, gamma_kl_di[:, :, c_col]-gamma_kl_sum_di, precision='highest') - gamma_h_di

    return out

@jax.jit
def update_phi_h_mis(phi_g, gamma_kl_di, gamma_kl_sum_di, gamma_h_di, c_col, c_col_mask):
    """q(h_j | phi^h)."""
    # phi_tmp = phi_g.transpose(1,0)[:, jnp.newaxis, :]
    # tmp1 = gamma_kl_di[:, :, c_col]
    # inner_sum = (phi_tmp*(tmp1-gamma_kl_sum_di)).sum(axis=0)
    # out = (c_col_mask[jnp.newaxis,:] * inner_sum).sum(axis=1) - gamma_h_di
    # # tmp_mask = (c_col != -1)[jnp.newaxis, jnp.newaxis,:]
    # # out = (phi_tmp*(tmp1-gamma_kl_sum_di)*tmp_mask).sum(axis=(0,2)) - gamma_h_di

    inner_sum = jnp.einsum("nk, kln -> ln", phi_g, gamma_kl_di[:, :, c_col]-gamma_kl_sum_di, precision='highest')
    out = (jnp.einsum("n,ln -> l", c_col_mask, inner_sum, precision='highest') - gamma_h_di)

    return out

vmap_update_phi_h_full = jax.vmap(update_phi_h_full, in_axes=(None, None, None, None, 1))
vmap_update_phi_h_mis = jax.vmap(update_phi_h_mis, in_axes=(None, None, None, None, 1, 1))
jax.jit(vmap_update_phi_h_full)
jax.jit(vmap_update_phi_h_mis)

def update_phi_g(phi_h, gamma_kl_di, gamma_kl_sum_di, gamma_g_di, C, C_mask=None, missing=False):
    if not missing:
        return vmap_update_phi_g_full(phi_h, gamma_kl_di, gamma_kl_sum_di, gamma_g_di, C)
    else:
        assert C_mask is not None
        return vmap_update_phi_g_mis(phi_h, gamma_kl_di, gamma_kl_sum_di, gamma_g_di, C, C_mask)
    
def update_phi_h(phi_g, gamma_kl_di, gamma_kl_sum_di, gamma_h_di, C, C_mask=None, missing=False):
    if not missing:
        return vmap_update_phi_h_full(phi_g, gamma_kl_di, gamma_kl_sum_di, gamma_h_di, C)
    else:
        assert C_mask is not None
        return vmap_update_phi_h_mis(phi_g, gamma_kl_di, gamma_kl_sum_di, gamma_h_di, C, C_mask)

################################################
# Global update gamma_g
@jax.jit
def update_gamma_g(alpha_g, phi_g, update_factor=1):
    """Update Dirichlet parameters for q(pi^g)."""
    return alpha_g + update_factor*phi_g.sum(axis=0)

# Global update gamma_h
@jax.jit
def update_gamma_h(alpha_h, phi_h, update_factor=1):
    """Update Dirichlet parameters for q(pi^h)."""
    return alpha_h + update_factor*phi_h.sum(axis=0)

################################################
# Global update gamma_kl (original)
@partial(jax.jit, static_argnums=(4,5,6))
def update_gamma_kl_1h(alpha, phi_g, phi_h, C_1h, K, L, num_categories=13, update_factor=1):
    """Update Dirichlet parameters for q(pi^(k,l))."""
    gamma_kl = alpha * jnp.ones((K, L, num_categories))
    for k in range(K):
        for l in range(L):
            # # get outer product matrix (N x M x 1)
            # phi_outer = jnp.outer(phi_g[:, k], phi_h[:, l])[..., jnp.newaxis]
            # # Multiply by (N x M x 13) one hot vector and sum together into 13-dim vector
            # tmp_kl = (phi_outer * C_1h).sum(axis=(0,1))
            # # Update the kl entry
            # gamma_kl = gamma_kl.at[k,l,:].set(alpha + update_factor*tmp_kl)

            phi_outer = jnp.outer(phi_g[:, k], phi_h[:, l])
            tmp_kl = jnp.einsum("nm, nmc -> c", phi_outer, C_1h, precision='highest')
            gamma_kl = gamma_kl.at[k,l,:].set(alpha + update_factor*tmp_kl)

    return gamma_kl

def update_gamma_kl_ind(alpha, phi_g, phi_h, C, K, L, num_categories=13, update_factor=1):
    """Update Dirichlet parameters for q(pi^(k,l)) without using a one-hot encoded matrix."""
    gamma_kl = alpha * jnp.ones((K, L, num_categories))
    
    # Sum contributions for each category
    for cat_select in range(num_categories):
        # Boolean mask for selected category
        cat_mask = (C == cat_select)
        tmp_kl_cat = update_gamma_kl_ind_cat(alpha, phi_g, phi_h, cat_mask, K, L, update_factor=update_factor)
        gamma_kl = gamma_kl.at[:, :, cat_select].set(tmp_kl_cat)

    return gamma_kl

@partial(jax.jit, static_argnums=(4,5,6))
def update_gamma_kl_ind_cat(alpha, phi_g, phi_h, cat_mask, K, L, update_factor=1):
    """Update Dirichlet parameters for q(pi^(k,l)) without using a one-hot encoded matrix."""
    # assert C_mask.shape == (C_1h.shape[0], C_1h.shape[1], 1)

    gamma_kl = alpha * jnp.ones((K, L))    
    # Sum contributions for each category
    for k in range(K):
        for l in range(L):
            # Get outer product matrix (N x M)
            phi_outer = jnp.outer(phi_g[:, k], phi_h[:, l])
            tmp_kl_cat = (phi_outer * cat_mask).sum()
            gamma_kl = gamma_kl.at[k, l].set(alpha + update_factor * tmp_kl_cat)

    return gamma_kl

################################################
@partial(jax.jit, static_argnums=(5,6,7,8))
def update_gamma_kl_1h_mis(alpha, phi_g, phi_h, C_1h, C_mask, K, L, num_categories=13, update_factor=1):
    """Update Dirichlet parameters for q(pi^(k,l))."""
    # assert C_mask.shape == (C_1h.shape[0], C_1h.shape[1], 1)
    # C_mask = C_mask[..., jnp.newaxis]
    # out_kl = jnp.einsum("nm, nk, ml, nmc -> klc", C_mask, phi_g, phi_h, C_1h)

    gamma_kl = alpha * jnp.ones((K, L, num_categories))
    for k in range(K):
        for l in range(L):
            phi_outer = jnp.outer(phi_g[:, k], phi_h[:, l])
            tmp_kl = jnp.einsum("nm, nm, nmc -> c", C_mask, phi_outer, C_1h, precision='highest')
            gamma_kl = gamma_kl.at[k,l,:].set(alpha + update_factor*tmp_kl)

    return gamma_kl

def update_gamma_kl_ind_mis(alpha, phi_g, phi_h, C, C_mask, K, L, num_categories=13, update_factor=1):
    """Update Dirichlet parameters for q(pi^(k,l)) without using a one-hot encoded matrix."""
    gamma_kl = alpha * jnp.ones((K, L, num_categories))
    
    # Sum contributions for each category
    for cat_select in range(num_categories):
        # Boolean mask for selected category
        cat_mask = (C == cat_select)
        tmp_kl_cat = update_gamma_kl_ind_mis_cat(alpha, phi_g, phi_h, cat_mask, C_mask, K, L, update_factor=update_factor)
        gamma_kl = gamma_kl.at[:, :, cat_select].set(tmp_kl_cat)

    return gamma_kl

@partial(jax.jit, static_argnums=(5,6,7))
def update_gamma_kl_ind_mis_cat(alpha, phi_g, phi_h, cat_mask, C_mask, K, L, update_factor=1):
    """Update Dirichlet parameters for q(pi^(k,l)) without using a one-hot encoded matrix."""
    # assert C_mask.shape == (C_1h.shape[0], C_1h.shape[1], 1)

    gamma_kl = alpha * jnp.ones((K, L))    
    # Sum contributions for each category
    for k in range(K):
        for l in range(L):
            # Get outer product matrix (N x M)
            phi_outer = jnp.outer(phi_g[:, k], phi_h[:, l])
            tmp_kl_cat = (C_mask*phi_outer * cat_mask).sum()
            gamma_kl = gamma_kl.at[k, l].set(alpha + update_factor * tmp_kl_cat)

    return gamma_kl

################################################

def update_gamma_kl(alpha, phi_g, phi_h, C, K, L, C_mask=None, C_1h=None, num_categories=13, update_factor=1, update_ind=True, missing=False):
    if not missing:
        if not update_ind:
            gamma_kl = update_gamma_kl_1h(alpha, phi_g, phi_h, C_1h, K, L, num_categories, update_factor=update_factor)
        else:
            gamma_kl = update_gamma_kl_ind(alpha, phi_g, phi_h, C, K, L, num_categories, update_factor=update_factor)
    else:
        if not update_ind:
            gamma_kl = update_gamma_kl_1h_mis(alpha, phi_g, phi_h, C_1h, C_mask, K, L, num_categories, update_factor=1)
        else:
            gamma_kl = update_gamma_kl_ind_mis(alpha, phi_g, phi_h, C, C_mask, K, L, num_categories, update_factor=update_factor)

    return gamma_kl

################################################

"""ELBO calculation functions"""
# functions to compute ELBO components
@partial(jax.jit, static_argnums=(2))
def KLD_dirichlet(alpha_q, alpha_p, axis=None):
    assert axis is None or axis == 2
    # params q~alpha_q, p~alpha_p (1 dirichlet distribution)
    alpha_p = jnp.full_like(alpha_q, alpha_p) if jnp.ndim(alpha_p) == 0 else alpha_p

    # precompute sum
    alpha_q_0 = alpha_q.sum(axis=axis)
    alpha_p_0 = alpha_p.sum(axis=axis)

    # Compute the first term: log B(alpha_q) - log B(alpha_p)
    log_B_alpha_q = gammaln(alpha_q_0) - gammaln(alpha_q).sum(axis=axis)
    log_B_alpha_p = gammaln(alpha_p_0) - gammaln(alpha_p).sum(axis=axis)
    log_B_ratio = log_B_alpha_q - log_B_alpha_p

    # Compute the second term: (alpha_q_k - alpha_p_k) * (psi(alpha_q_k) - psi(alpha_q_0))
    digamma_alpha_q = digamma(alpha_q)
    digamma_alpha_q_0 = digamma(alpha_q_0) if axis == None else digamma(alpha_q_0)[:, :, jnp.newaxis]
    second_term = jnp.sum((alpha_q - alpha_p) * (digamma_alpha_q - digamma_alpha_q_0), axis=axis)

    # KL divergence
    kl = log_B_ratio + second_term

    return kl

@jax.jit
def categorical_entropy(p):
    return entr(p).sum(axis=1)

@jax.jit
def E_log_dirichlet(pi):
    digamma_pi = digamma(pi)
    digamma_pi_0 = digamma(pi.sum())

    return digamma_pi - digamma_pi_0

@jax.jit
def KLD_gpi(cat_q, pi_q, pi_p):
    # pi_q, pi_p are posterior and prior dirichlet
    # cat_q is phi_g (posterior categorical assignment, no need for prior categorical?)

    # term 1
    kld = KLD_dirichlet(pi_q, pi_p)

    # term 2
    cat_entr = categorical_entropy(cat_q)

    E_log_gamma_g = E_log_dirichlet(pi_q)[jnp.newaxis,:]
    tmp = (cat_q*E_log_gamma_g).sum(axis=1)

    term2 = (cat_entr - tmp).sum()
    # print(kld, term2)

    return kld + term2

################################################

def loglik_q(C, phi_g, phi_h, gamma_kl, C_mask=None, missing=False):
    if not missing:
        return loglik_q_full(C, phi_g, phi_h, gamma_kl)
    else:
        assert C_mask is not None
        return loglik_q_mis(C, C_mask, phi_g, phi_h, gamma_kl)

@jax.jit
def loglik_q_full(C, phi_g, phi_h, gamma_kl):
    K, L, _ = gamma_kl.shape
    assert phi_g.shape[1] == K and phi_h.shape[1] == L

    digamma_kl_0 = digamma(gamma_kl.sum(axis=2))
    digamma_kl = digamma(gamma_kl)
    # digamma_kl_full = digamma(gamma_kl)[:, :, C]
    
    ll = 0.0
    for k in range(K):
        for l in range(L):
            # digamma_kl = digamma(gamma_kl)[k, l, C]

            phi_outer = jnp.outer(phi_g[:, k], phi_h[:, l])
            tmp = digamma_kl[k, l, C] - digamma_kl_0[k,l]
            # tmp = digamma_kl_full[k,l] - digamma_kl_0[k,l]

            ll += (phi_outer*tmp).sum()

    return ll

@jax.jit
def loglik_q_mis(C, C_mask, phi_g, phi_h, gamma_kl):
    K, L, _ = gamma_kl.shape
    assert phi_g.shape[1] == K and phi_h.shape[1] == L

    digamma_kl_0 = digamma(gamma_kl.sum(axis=2))
    digamma_kl = digamma(gamma_kl)
    # digamma_kl_full = digamma(gamma_kl)[:, :, C]
    
    ll = 0.0
    for k in range(K):
        for l in range(L):
            # phi_outer = C_mask*jnp.outer(phi_g[:, k], phi_h[:, l])
            phi_outer = C_mask*jnp.exp(jnp.log(phi_g[:, k] + 1e-10)[:, jnp.newaxis] + jnp.log(phi_h[:, l] + 1e-10))
            tmp = digamma_kl[k, l, C] - digamma_kl_0[k,l]

            ll += (phi_outer*tmp).sum()

    return ll

loglik_q_full1 = functools.partial(loglik_q_full)
loglik_q_mis1 = functools.partial(loglik_q_mis)

def loglik_q1(C, phi_g, phi_h, gamma_kl, C_mask=None, missing=False):
    if not missing:
        return loglik_q_full1(C, phi_g, phi_h, gamma_kl)
    else:
        assert C_mask is not None
        return loglik_q_mis1(C, C_mask, phi_g, phi_h, gamma_kl)

"""Likelihood and posterior functions"""
@jax.jit
def post_dir(dir_params):
    # use posterior dirichlet distribution and get mean/variance
    dir_sum = dir_params.sum(axis=2)[:, :, jnp.newaxis]

    dir_mean = dir_params/dir_sum
    dir_variance = dir_mean*(1-dir_mean)/(1+dir_sum)

    return dir_mean, dir_variance

@partial(jax.jit, static_argnums=(1))
def log_post_dir(dir_params, tol=1e-7):
    # use posterior dirichlet distribution and get mean/variance
    dir_sum = dir_params.sum(axis=2)[:, :, jnp.newaxis]

    log_dir_mean = jnp.log(dir_params) - jnp.log(dir_sum)
    log_dir_variance = log_dir_mean + jnp.log(1-jnp.exp(log_dir_mean + tol)) - jnp.log(1+dir_sum)

    return log_dir_mean, log_dir_variance

def sbm_log_lik(C, log_probs, g, h, reduction='sum', C_mask=None):
    # probs: multinomial probs
    # C data lookup matrix | g, h are cluster assignments for rows and columns

    # get matrix of cluster distributions based on cluster assignments
    tmp = log_probs[jnp.ix_(g, h)]

    # Generate row and column indices
    rows, cols = jnp.meshgrid(jnp.arange(C.shape[0]), jnp.arange(C.shape[1]), indexing='ij')

    # Extract values
    result = tmp[rows, cols, C] if C_mask is None else C_mask*tmp[rows, cols, C]
    if reduction == 'sum':
        return result.sum()
    else:
        return result
    
def sbm_log_lik_slow(C, log_probs, g, h, num_cat, C_mask=None):
    # probs: multinomial probs
    # C data lookup matrix | g, h are cluster assignments for rows and columns

    # get matrix of cluster distributions based on cluster assignments
    g_unique, h_unique = jnp.unique(g), jnp.unique(h)
    out = 0
    for k in g_unique:
        for l in h_unique:
            idx = jnp.argwhere(g == k).squeeze(1)
            idy = jnp.argwhere(h == l).squeeze(1)
            sub_C = C[jnp.ix_(idx, idy)]
            sub_C_mask = C_mask[jnp.ix_(idx, idy)] if C_mask is not None else None
            
            for c in range(num_cat):
                count = (sub_C == c).sum() if C_mask is None else (sub_C_mask*(sub_C == c)).sum()
                out += count*log_probs[k, l, c]

    return out

def compute_cluster_ll(labels, eps=1e-12):
    labels = np.array(labels)
    n = len(labels)
    num_clusters = np.max(labels)+1
    
    # Count occurrences of each label from 0 to num_clusters - 1
    counts = np.bincount(labels, minlength=num_clusters)
    
    # Empirical probabilities (MLE)
    probs = counts / n
    
    # Filter out zero counts to avoid log(0)
    nonzero_counts = counts[counts > 0]
    nonzero_probs = probs[counts > 0]
    
    # Log-likelihood: sum_k count_k * log(prob_k)
    log_likelihood = np.sum(nonzero_counts * np.log(nonzero_probs+eps))
    
    return log_likelihood

################################################

"""Plotting Functions"""
def plt_blocks(C, g_labels, h_labels, cluster_means, title='', print_labels=False):
    # Define the custom colors for categories
    copy_number_colors = [
        'yellow',          # -1
        '#377eb8',         # 0
        '#89c8e5',         # 1
        '#b3b3b3',         # 2
        '#fdb462',         # 3
        '#ff7f00',         # 4
        '#e41a1c',         # 5
        '#b2182b',         # 6
        '#d6604d',         # 7
        '#ce1256',         # 8
        '#df65b0',         # 9
        '#d4b9da',         # 10
        '#762a83',         # 11
        '#762a99'          # 12
    ]
    
    # Create a colormap for the custom colors
    custom_cmap = ListedColormap(copy_number_colors)
    norm = BoundaryNorm(boundaries=np.arange(-1, 12), ncolors=len(copy_number_colors))

    # Process the input matrix for block reordering
    C_out = (
        pd.DataFrame(C)
        .assign(g=np.array(g_labels)).sort_values('g')
        .transpose()
        .assign(h=np.array(list(np.array(h_labels)) + [max(g_labels) + 1])).sort_values('h')
        .transpose()
    )

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Original heatmap
    sns.heatmap(C, ax=ax[0], cmap=custom_cmap, cbar=False, norm=norm)
    # Block-reordered heatmap
    sns.heatmap(C_out.drop('h').drop(columns='g'), ax=ax[1], cmap=custom_cmap, cbar=False, norm=norm)
    # Cluster means heatmap
    sns.heatmap(pd.DataFrame(cluster_means), ax=ax[2], cmap=custom_cmap, norm=norm)
    
    ax[0].set(title='Original')
    ax[1].set(title='Block-reordered' + title)
    ax[2].set(title='Multinomial Argmax' + title)

    # Optional: Print labels
    if print_labels:
        print("Cells", g_labels)
        print("Bins", h_labels)

    plt.tight_layout()
    plt.show()

# perform independent bi-clustering
def sc_clusters(C, K, L, random_state=0):
    sc_g = SpectralClustering(
        n_clusters=K,
        assign_labels='discretize',
        random_state=random_state
    ).fit(C).labels_

    sc_h = SpectralClustering(
        n_clusters=L,
        assign_labels='discretize',
        random_state=random_state
    ).fit(C.transpose()).labels_

    return sc_g, sc_h

# Compute individual cluster proportions
def cluster_proportions(C, K, L, num_cat, g, h):
    proportions = np.zeros((K, L, num_cat), dtype=float)

    for k in range(K):
        for l in range(L):
            # Mask for rows and columns belonging to clusters (k, l)
            mask = np.outer(g == k, h == l)
            C_subset = C[mask]  # Extract relevant values

            # Count occurrences of each category
            counts = np.bincount(C_subset, minlength=num_cat)

            # Normalize to proportions
            total = counts.sum()
            if total > 0:
                proportions[k, l] = counts / total

    return proportions

# get initializations for rows and columns
def sc_init(C, K, L, num_cat, random_state=0):
    sc_g, sc_h = sc_clusters(C, K, L, random_state)
    proportions = cluster_proportions(C, K, L, num_cat, sc_g, sc_h)

    return sc_g, sc_h, proportions

def sc_bi_init(C, K, L, num_cat, random_state=0, method="log"):
    model = SpectralBiclustering(n_clusters=(K,L), method=method, random_state=random_state)
    model.fit(C)
    sc_g, sc_h = model.row_labels_, model.column_labels_

    proportions = cluster_proportions(C, K, L, num_cat, sc_g, sc_h)

    return sc_g, sc_h, proportions

# Define the probability vectors according to predefined clusters
# def generate_phi(N, num_cat, clusters, concentration=0.9):
#     """
#     Generate a probability tensor of shape (K, L, num_cat)
#     with most mass on one category using a Dirichlet distribution.
#     """
#     # Initialize probability tensor
#     phi = np.zeros((N, num_cat))

#     for n in range(N):
#         # Generate a vector where one category dominates
#         dominant = clusters[n]
#         dirichlet_weights = np.full(num_cat, (1 - concentration) / (num_cat - 1))
#         dirichlet_weights[dominant] = concentration

#         # Sample from a Dirichlet centered around dirichlet_weights
#         phi[n, :] = np.random.dirichlet(dirichlet_weights * 100)

#     return phi


def estimate_obs_propensity(C, mode='prod'):
    """
    C: np.ndarray of shape (N, M), possibly containing np.nan for missing values.

    Returns:
        propensity_matrix: np.ndarray of shape (N, M) with estimated probabilities,
                           with 0 at the missing entries in C.
    """
    assert mode in ['prod', 'sum'], 'Only implemented for mode = prod or sum'
    # Identify missing values
    missing_mask = np.array(C != -1)

    # Calculate row and column missing rates
    row_missing_rate = np.mean(missing_mask, axis=1, keepdims=True)  # shape: (N, 1)
    col_missing_rate = np.mean(missing_mask, axis=0, keepdims=True)  # shape: (1, M)

    # Combine row and column effects
    if mode == 'prod':
        propensity = (row_missing_rate + col_missing_rate) / 2
    elif mode == 'sum':
        propensity = (row_missing_rate*col_missing_rate)

    propensity = 1/propensity

    # Set missing entries' propensity to 0
    propensity[missing_mask==False] = 0

    return propensity

@partial(jax.jit, static_argnums=(0,1))
def generate_phi_g(N, num_cat, clusters, concentration=0.9, key=None):
    if key is None:
        key = jrand.PRNGKey(np.random.randint(0, 1e6))  # Generate a random key if none is provided

    # Compute Dirichlet weights
    base_weights = jnp.full((N, num_cat), (1 - concentration) / (num_cat - 1))
    base_weights = base_weights.at[jnp.arange(N), clusters].set(concentration)

    # Generate random keys for each row
    keys = jrand.split(key, N)

    # Sample Dirichlet in parallel
    phi = jax.vmap(lambda k, w: jrand.dirichlet(k, w * 100))(keys, base_weights)

    return phi

@partial(jax.jit, static_argnums=(0,1))
def generate_phi_h(N, num_cat, clusters, concentration=0.9, key=None):
    if key is None:
        key = jrand.PRNGKey(np.random.randint(0, 1e6))  # Generate a random key if none is provided

    # Compute Dirichlet weights
    base_weights = jnp.full((N, num_cat), (1 - concentration) / (num_cat - 1))
    base_weights = base_weights.at[jnp.arange(N), clusters].set(concentration)

    # Generate random keys for each row
    keys = jrand.split(key, N)

    # Sample Dirichlet in parallel
    phi = jax.vmap(lambda k, w: jrand.dirichlet(k, w * 100))(keys, base_weights)

    return phi

def ICL_penalty(N, M, K_unique, L_unique, num_cat, ICL_factor=0.5):
    nr_edges = jnp.log(N) + jnp.log(M)
    nr_connectivities = K_unique*L_unique
    dim_connectivities = num_cat - 1
    
    edge_penalty = dim_connectivities*nr_connectivities*nr_edges
    block_penalty = (K_unique-1)*jnp.log(N) + (L_unique-1)*jnp.log(M)

    return ICL_factor*(block_penalty + edge_penalty)

def jax_array_to_csv(array, output_path):
    """
    Convert a 3D JAX array (KxLx13) to a CSV file with columns for K, L, and the 13 features.
    
    Parameters:
    array: jax.numpy.ndarray of shape (K, L, 13)
    output_path: str, path where the CSV file should be saved
    """
    # Get dimensions
    K, L, num_cat = array.shape
    
    # Create meshgrid for K and L indices
    k_indices, l_indices = jnp.meshgrid(jnp.arange(K), jnp.arange(L), indexing='ij')
    

    # Reshape arrays
    k_flat = k_indices.flatten()
    l_flat = l_indices.flatten()
    features_flat = array.reshape(-1, num_cat)
    
    # Create column names
    feature_cols = [f'cat_{i}' for i in range(num_cat)]
    
    # Create DataFrame
    df = pd.DataFrame(features_flat, columns=feature_cols)
    df.insert(0, 'K', k_flat)
    df.insert(1, 'L', l_flat)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df