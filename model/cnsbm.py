from .utils_cnsbm import (
    normalize_log_probs, 
    KLD_dirichlet, KLD_gpi, post_dir, log_post_dir,
    plt_blocks,
    cluster_proportions, sc_init, sc_bi_init, estimate_obs_propensity,
    generate_phi_g, generate_phi_h, ICL_penalty, jax_array_to_csv,
    compute_cluster_ll
)
from . import utils_cnsbm
import os, time, gc, pickle, functools
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from collections import deque

def robbins_monro_schedule(t, eta_0=0.01, tau=5000, kappa=0.75):
    return eta_0 / (1 + t / tau) ** kappa

def svi_schedule(t, tau=1, kappa=0.9):
    return (t + tau) ** (-kappa)

"""SBM Model"""
class CNSBM:
    def __init__(self, C, K, L, alphas=None, phis=None, gammas=None,
                 init_clusters=None, rand_init='random', target_cats=None, target_concentration=None, 
                 concentration=0.9, warm_start=False, fill_na=0, propensity_mode=None, seed=42):
        # Rng
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.seeds = self.rng.integers(low=0, high=2**32, size=100, dtype=np.uint32)
        
        # Matrix Size
        self.C = C
        self.N, self.M = self.C.shape
        self.num_cat = int(C.max() + 1)
        self.fill_na = fill_na
        self.propensity_mode = propensity_mode

        if not (self.C != -1).all():
            print(f"Detecting missing values - will use missingness mask (mode {propensity_mode}) and fill missing values with {fill_na}")
            self.missing = True
            self.C = jnp.where(self.C == -1, self.fill_na, self.C)
            if self.propensity_mode is None:
                self.C_mask = (self.C != -1)
            else:
                self.C_mask = jnp.asarray(estimate_obs_propensity(np.array(self.C), mode=self.propensity_mode))
        else:
            self.C_mask = None
            self.missing = False

        # Initialize update functions
        self.update_phi_g = utils_cnsbm.update_phi_g
        self.update_phi_h = utils_cnsbm.update_phi_h
        self.update_gamma_kl = utils_cnsbm.update_gamma_kl
        self.loglik_q = utils_cnsbm.loglik_q
        self.KLD_gpi = functools.partial(KLD_gpi) #reuse function
        self.KLD_hpi = functools.partial(KLD_gpi)
        self.sbm_log_lik = utils_cnsbm.sbm_log_lik
        self.sbm_log_lik_slow = utils_cnsbm.sbm_log_lik_slow

        # for batched elbo computation when using stochastic VI
        self.loglik_q1 = utils_cnsbm.loglik_q1
        self.KLD_gpi1 = functools.partial(KLD_gpi)
        self.KLD_hpi1 = functools.partial(KLD_gpi)

        # Cluster dimensions
        self.K, self.L = K, L

        # SBM object
        self.rand_init = rand_init
        self.fitted = False

        # Training history (store iteration metrics)
        self.training_history = []
        self.ICL_fitted = {}

        print(f"Initializing variational distributions (seed {self.seed})...")

        # If using spectral clustering initialization
        if self.rand_init == 'spectral':
            self.init_g, self.init_h, self.init_proportions = sc_init(self.C, self.K, self.L, self.num_cat, self.seeds[0])
        elif self.rand_init == 'spectral_bi':
            self.init_g, self.init_h, self.init_proportions = sc_bi_init(self.C, self.K, self.L, self.num_cat, self.seeds[1])
        elif self.rand_init == 'init':
            self.init_g, self.init_h = init_clusters['g'], init_clusters['h']
            if gammas is None and not warm_start:
                # when initializing a new model with initial values based on cluster proportions
                self.init_proportions = cluster_proportions(self.C, self.K, self.L, self.num_cat, self.init_g, self.init_h)
            elif gammas is not None and warm_start:
                # when using existing gammas as a warm start
                self.init_proportions = gammas['gamma_kl'] - 1
            else:
                # when using warm start based on cluster labels (self.init_proportions will be skipped by using update_gammas_warm)
                rng = np.random.default_rng(self.seeds[2])
                self.init_proportions = rng.dirichlet(alpha=np.ones(self.num_cat), size=(self.K, self.L))  # Shape (K, L, num_cat)
                # self.init_proportions = np.random.dirichlet(alpha=np.ones(self.num_cat), size=(self.K, self.L))  # Shape (K, L, num_cat)

        # Initialize priors
        if alphas is None:
            # uniform prior over the categories
            self.alpha_g, self.alpha_h, self.alpha_pi = 1,1,1
        else:
            self.alphas = alphas
            self.alpha_g, self.alpha_h, self.alpha_pi = alphas['alpha_g'], alphas['alpha_h'], alphas['alpha_pi']

        if phis is None:
            if self.rand_init == 'uniform':
                # uniform over the cluster assignments
                self.phi_g = jnp.full((self.N, self.K), 1.0 / self.K)  # q(g_i)
                self.phi_h = jnp.full((self.M, self.L), 1.0 / self.L)  # q(h_j)
            elif self.rand_init in ['random', 'random_target']:
                rng = np.random.default_rng(self.seeds[3])
                rng1 = np.random.default_rng(self.seeds[4])
                self.phi_g = jnp.asarray(rng.dirichlet(alpha=np.ones(self.K), size=self.N))  # Shape (N, K)
                self.phi_h = jnp.asarray(rng1.dirichlet(alpha=np.ones(self.L), size=self.M))  # Shape (M, L)
            elif self.rand_init in ['spectral', 'spectral_bi', 'init']:
                self.phi_g = generate_phi_g(self.N, self.K, self.init_g, concentration=concentration, key=jax.random.PRNGKey(self.seeds[5]))
                self.phi_h = generate_phi_h(self.M, self.L, self.init_h, concentration=concentration, key=jax.random.PRNGKey(self.seeds[6]))
        else:
            self.phis = phis
            self.phi_g, self.phi_h = phis['phi_g'], phis['phi_h']

        if gammas is None:
            rng = np.random.default_rng(self.seeds[7])
            rng1 = np.random.default_rng(self.seeds[8])
            rng2 = np.random.default_rng(self.seeds[9])
            if self.rand_init == 'uniform':
                # uniform over the cluster probabilities and block distributions
                self.gamma_g = jnp.ones(self.K)    # q(pi^g)
                self.gamma_h = jnp.ones(self.L)    # q(pi^h)
                self.gamma_kl = jnp.ones((self.K, self.L, self.num_cat))  # q(pi^(k,l))
            elif self.rand_init == 'random':
                # 2. Initialize gamma_g and gamma_h as positive values
                self.gamma_g = rng.gamma(shape=2.0, scale=1.0, size=self.K)  # Shape (K,)
                self.gamma_h = rng1.gamma(shape=2.0, scale=1.0, size=self.L)  # Shape (L,)
                self.gamma_kl = rng2.dirichlet(alpha=np.ones(self.num_cat), size=(self.K, self.L))  # Shape (K, L, num_cat)
            elif self.rand_init == 'random_target':
                assert target_cats is not None and target_concentration is not None
                # 2. Initialize gamma_g and gamma_h as positive values
                self.gamma_g = rng.gamma(shape=2.0, scale=1.0, size=self.K)  # Shape (K,)
                self.gamma_h = rng1.gamma(shape=2.0, scale=1.0, size=self.L)  # Shape (L,)
                
                alpha_kl = np.ones(self.num_cat)  # Base concentration
                alpha_kl[target_cats] += target_concentration  # Add weight to target categories
                self.gamma_kl = rng2.dirichlet(alpha=alpha_kl, size=(self.K, self.L))  # Shape (K, L, num_cat)
            elif self.rand_init in ['spectral', 'spectral_bi', 'init']:
                self.gamma_g = rng.gamma(shape=2.0, scale=1.0, size=self.K)  # Shape (K,)
                self.gamma_h = rng1.gamma(shape=2.0, scale=1.0, size=self.L)  # Shape (L,)
                self.gamma_kl = 1 + self.init_proportions
        else:
            self.gammas = gammas
            self.gamma_g, self.gamma_h, self.gamma_kl = gammas['gamma_g'], gammas['gamma_h'], gammas['gamma_kl']

    def update_gammas_warm(self, update_ind=False):
        print("Setting warm start updates for gammas...")
        if not update_ind:
            self.C_1h = jax.nn.one_hot(self.C, num_classes=self.num_cat)
        else:
            self.C_1h = None

        self.gamma_kl = self.update_gamma_kl(self.alpha_pi, self.phi_g, self.phi_h, self.C, self.K, self.L, self.C_mask, self.C_1h, self.num_cat, update_factor=1, update_ind=update_ind, missing=self.missing)
        self.gamma_g = utils_cnsbm.update_gamma_g(self.alpha_g, self.phi_g, update_factor=1)
        self.gamma_h = utils_cnsbm.update_gamma_h(self.alpha_h, self.phi_h, update_factor=1)

    def predict_phi_g(self, C_new):
        assert C_new.shape[1] == self.M
        C_new_mask = (C_new != -1)
        if not C_new_mask.all():
            print("Detecting missing values - will use missingness mask")
            missing = True
        else:
            C_new_mask = None
            missing = False

        if self.fitted:
            phi_h, gamma_g, gamma_kl = (self.fitted_params[keys] for keys in ('phi_h', 'gamma_g', 'gamma_kl'))
        else:
            phi_h, gamma_g, gamma_kl = self.phi_h, self.gamma_g, self.gamma_kl

        # Local updates
        gamma_g_di = digamma(gamma_g)
        gamma_kl_sum_di = digamma(gamma_kl.sum(axis=2)[:, :, jnp.newaxis])
        gamma_kl_di = digamma(gamma_kl)

        phi_g = normalize_log_probs(self.update_phi_g(phi_h, gamma_kl_di, gamma_kl_sum_di, gamma_g_di, C_new, C_new_mask, missing))

        return phi_g

    def batch_vi(self, num_iters, tol=1e-6, batch_print=50, fitted=False, update_ind=True, window_size=5):
        # whether to use 1h encoded matrix
        self.C_1h = jax.nn.one_hot(jnp.where(self.C == -1, self.fill_na, self.C), num_classes=self.num_cat) if not update_ind else None
        
        if fitted:
            phi_g, phi_h, gamma_g, gamma_h, gamma_kl = (self.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl'))
        else:
            phi_g, phi_h = self.phi_g, self.phi_h
            gamma_g, gamma_h, gamma_kl = self.gamma_g, self.gamma_h, self.gamma_kl

        print("Running batch variational inference...")

        # Run VI loop
        # elbo_prev = -jnp.inf
        elbo_history = deque(maxlen=window_size)
        elbo_prev_avg = None
        for i in range(num_iters):
            start_time = time.time()
            # Local updates
            gamma_g_di, gamma_h_di = digamma(gamma_g), digamma(gamma_h)
            gamma_kl_sum_di = digamma(gamma_kl.sum(axis=2)[:, :, jnp.newaxis])
            gamma_kl_di = digamma(gamma_kl)
            phi_g = normalize_log_probs(self.update_phi_g(phi_h, gamma_kl_di, gamma_kl_sum_di, gamma_g_di, self.C, self.C_mask, self.missing))
            phi_h = normalize_log_probs(self.update_phi_h(phi_g, gamma_kl_di, gamma_kl_sum_di, gamma_h_di, self.C, self.C_mask, self.missing))

            # Global updates
            gamma_g = utils_cnsbm.update_gamma_g(self.alpha_g, phi_g, update_factor=1)
            gamma_h = utils_cnsbm.update_gamma_h(self.alpha_h, phi_h, update_factor=1)
            gamma_kl = self.update_gamma_kl(self.alpha_pi, phi_g, phi_h, self.C, self.K, self.L, self.C_mask, self.C_1h, self.num_cat, update_factor=1, update_ind=update_ind, missing=self.missing)

            elbo, ll, KL_g, KL_h, KL_kl = self.elbo(phi_g, phi_h, gamma_g, gamma_h, gamma_kl, fitted=False, verbose=False)
            end_time = time.time()

            if i == 0:
                self.compilation_time = end_time - start_time

            # Save iteration results
            self.training_history.append({
                "iteration": i, "ELBO": elbo,
                "LogLik": ll, "KL-g": KL_g, "KL-h": KL_h, "KL-kl": KL_kl
            })

            elbo_history.append(elbo)
            # Wait until the buffer is full
            if len(elbo_history) == window_size:
                elbo_avg = jnp.mean(jnp.array(elbo_history))
                
                if elbo_prev_avg is not None and jnp.abs(elbo_avg - elbo_prev_avg) < tol:
                    print(f'Iteration {i}, ELBO: {elbo:,.3f}, Loglik: {ll:,.3f}, KL-g: {KL_g:,.3f}, KL-h: {KL_h:,.3f}, KL-kl: {KL_kl:,.3f}')
                    print(f"ELBO converged at iteration {i}.")
                    break
                
                elbo_prev_avg = elbo_avg
            # # if jnp.abs(elbo - elbo_prev) < tol:
            # else:
            #     elbo_prev = elbo

            if i % batch_print == 0:
                print(f'Iteration {i}, ELBO: {elbo:,.3f}, Loglik: {ll:,.3f}, KL-g: {KL_g:,.3f}, KL-h: {KL_h:,.3f}, KL-kl: {KL_kl:,.3f}')
                print(f"Time elapsed: {end_time - start_time:,.2f} seconds")

        # posterior parameters
        self.fitted = True
        self.fitted_params = {'phi_g': phi_g, 'phi_h': phi_h, 'gamma_g': gamma_g, 'gamma_h': gamma_h, 'gamma_kl': gamma_kl}
        self.set_posteriors()

        return phi_g, phi_h, gamma_g, gamma_h, gamma_kl

    def stochastic_vi(self, num_iters, batch_size, local_max_iters=100, local_tol=0.0005,
                      eta_0=0.1, tau=5000, kappa=0.75, batches_per_epoch=100, tol=1e-4, batch_print=50,
                      fitted=False, update_ind=True, window_size=5, scheduler='svi',
                      elbo_batch_size=None):
        if fitted:
            phi_g, phi_h, gamma_g, gamma_h, gamma_kl = (self.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl'))
        else:
            phi_g, phi_h = self.phi_g, self.phi_h
            gamma_g, gamma_h, gamma_kl = self.gamma_g, self.gamma_h, self.gamma_kl
            
        # Get update factors and batch_size
        assert batch_size[0] <= self.N and batch_size[1] <= self.M
        update_factor_g = self.N/batch_size[0]
        update_factor_h = self.M/batch_size[1]
        update_factor_kl = self.N*self.M/(batch_size[0]*batch_size[1])

        print(f"Running stochastic variational inference... ({batches_per_epoch} batches per epoch)")

        # Copy initial parameters
        phi_g, phi_h = self.phi_g, self.phi_h
        gamma_g, gamma_h, gamma_kl = self.gamma_g, self.gamma_h, self.gamma_kl

        # Run VI loop
        # elbo_prev = -jnp.inf
        elbo_history = deque(maxlen=window_size)
        elbo_prev_avg = None
        for i in range(num_iters):
            start_time = time.time()

            local_iter_list = [0]*batches_per_epoch
            for batch_idx in range(batches_per_epoch):
                t = i * batches_per_epoch + batch_idx  # Cumulative iteration
                if scheduler=='svi':
                    eta_t = svi_schedule(t, tau, kappa)
                elif scheduler=='rb':
                    eta_t = robbins_monro_schedule(t, eta_0, tau, kappa)  # Compute learning rate

                # Batch subsample
                i_select = sorted(np.random.choice(self.N, size=batch_size[0], replace=False))
                j_select = sorted(np.random.choice(self.M, size=batch_size[1], replace=False))
                C_tmp = self.C[jnp.ix_(np.array(i_select), np.array(j_select))] if update_ind else None
                C_1h_tmp = jax.nn.one_hot(self.C[jnp.ix_(np.array(i_select), np.array(j_select))], num_classes=self.num_cat) if not update_ind else None
                Ci, Cj = self.C[i_select,:], self.C[:, j_select]
                if self.missing:
                    C_mask_tmp = self.C_mask[jnp.ix_(np.array(i_select), np.array(j_select))]
                    Ci_mask, Cj_mask = self.C_mask[i_select,:], self.C_mask[:, j_select]
                else:
                    C_mask_tmp, Ci_mask, Cj_mask = None, None, None
                
                # Local updates and initial local variables
                gamma_g_di, gamma_h_di = digamma(gamma_g), digamma(gamma_h)
                gamma_kl_sum_di = digamma(gamma_kl.sum(axis=2)[:, :, jnp.newaxis])
                gamma_kl_di = digamma(gamma_kl)

                int_phi_g, int_phi_h = phi_g[i_select,:], phi_h[j_select,:]

                local_converged = False
                for local_iter in range(local_max_iters):
                    new_phi_g = normalize_log_probs(self.update_phi_g(phi_h, gamma_kl_di, gamma_kl_sum_di, gamma_g_di, Ci, Ci_mask, self.missing))
                    phi_g = phi_g.at[i_select,:].set(new_phi_g)
                    
                    new_phi_h = normalize_log_probs(self.update_phi_h(phi_g, gamma_kl_di, gamma_kl_sum_di, gamma_h_di, Cj, Cj_mask, self.missing))
                    phi_h = phi_h.at[j_select,:].set(new_phi_h)

                    # Check differences
                    delta_phi_g = jnp.max(jnp.abs(new_phi_g - int_phi_g))
                    delta_phi_h = jnp.max(jnp.abs(new_phi_h - int_phi_h))

                    int_phi_g = new_phi_g
                    int_phi_h = new_phi_h

                    # check if update has converged
                    if delta_phi_g < local_tol and delta_phi_h < local_tol:
                        local_converged = True
                        break
                        
                # print(f"Local Iters: {local_iter}")
                local_iter_list[batch_idx] = local_iter
                if not local_converged and (i % batch_print == 0):
                    print(f"Local updates did not converge in {local_max_iters} iterations.")

                # intermediate global updates
                int_gamma_g = utils_cnsbm.update_gamma_g(self.alpha_g, int_phi_g, update_factor=update_factor_g)
                int_gamma_h = utils_cnsbm.update_gamma_h(self.alpha_h, int_phi_h, update_factor=update_factor_h)
                int_gamma_kl = self.update_gamma_kl(self.alpha_pi, int_phi_g, int_phi_h, C_tmp, self.K, self.L, C_mask_tmp, C_1h_tmp, self.num_cat, update_factor=update_factor_kl, update_ind=update_ind, missing=self.missing)

                # global updates
                gamma_g = (1-eta_t)*gamma_g + eta_t*int_gamma_g
                gamma_h = (1-eta_t)*gamma_h + eta_t*int_gamma_h
                gamma_kl = (1-eta_t)*gamma_kl + eta_t*int_gamma_kl

            elbo, ll, KL_g, KL_h, KL_kl = self.elbo(phi_g, phi_h, gamma_g, gamma_h, gamma_kl, fitted=False, verbose=False, elbo_batch_size=elbo_batch_size)
            end_time = time.time()

            if i == 0:
                self.compilation_time = end_time - start_time

            # Save iteration results
            self.training_history.append({
                "iteration": i, "ELBO": elbo,
                "LogLik": ll, "KL-g": KL_g, "KL-h": KL_h, "KL-kl": KL_kl
            })

            elbo_history.append(elbo)
            # Wait until the buffer is full
            if len(elbo_history) == window_size:
                elbo_avg = jnp.mean(jnp.array(elbo_history))
                
                if elbo_prev_avg is not None and jnp.abs(elbo_avg - elbo_prev_avg) < tol:
                    print(f'Iteration {i}, ELBO: {elbo:,.3f}, Loglik: {ll:,.3f}, KL-g: {KL_g:,.3f}, KL-h: {KL_h:,.3f}, KL-kl: {KL_kl:,.3f}')
                    print(f"ELBO converged at iteration {i}.")
                    break
                
                elbo_prev_avg = elbo_avg
            # if jnp.abs(elbo - elbo_prev) < tol:
            #     print(f'Iteration {i}, ELBO: {elbo:,.3f}, Loglik: {ll:,.3f}, KL-g: {KL_g:,.3f}, KL-h: {KL_h:,.3f}, KL-kl: {KL_kl:,.3f}')
            #     print(f"ELBO converged at iteration {i}.")
            #     break
            # else:
            #     elbo_prev = elbo

            if i % batch_print == 0:
                print("Local Updates:", local_iter_list)
                print(f'Iteration {i}, ELBO: {elbo:,.3f}, Loglik: {ll:,.3f}, KL-g: {KL_g:,.3f}, KL-h: {KL_h:,.3f}, KL-kl: {KL_kl:,.3f}')
                print(f"Time elapsed: {end_time - start_time:.2f} seconds; Learning Rate: {eta_t:.3f}")

        self.fitted = True
        self.fitted_params = {'phi_g': phi_g, 'phi_h': phi_h, 'gamma_g': gamma_g, 'gamma_h': gamma_h, 'gamma_kl': gamma_kl}
        self.set_posteriors()

        return phi_g, phi_h, gamma_g, gamma_h, gamma_kl

    def elbo(self, phi_g=None, phi_h=None, gamma_g=None, gamma_h=None, gamma_kl=None, fitted=False, verbose=False, elbo_batch_size=None):
        if fitted:
            assert self.fitted
            phi_g, phi_h, gamma_g, gamma_h, gamma_kl = (self.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl'))
        else:
            assert phi_g is not None and phi_h is not None
            assert gamma_g is not None and gamma_h is not None
            assert gamma_kl is not None

        # Compute all in one go (most efficient)
        if elbo_batch_size is None:
            ll = self.loglik_q(self.C, phi_g, phi_h, gamma_kl, self.C_mask, self.missing)
            KL_g = self.KLD_gpi(phi_g, gamma_g, self.alpha_g)
            KL_h = self.KLD_hpi(phi_h, gamma_h, self.alpha_h)
            KL_kl = KLD_dirichlet(gamma_kl, self.alpha_pi, axis=2).sum()
        else:
            ll, KL_g, KL_h, KL_kl = 0, 0, 0, 0
            batch_size_x, batch_size_y = elbo_batch_size
            
            # compute ll and KL_g (individual batches + last batch)
            batch_starts_x = jnp.arange(0, self.N, batch_size_x)
            for i in batch_starts_x[:-1]:
                mask = self.C_mask[i:(i+batch_size_x),:] if self.missing else None
                ll += self.loglik_q(self.C[i:(i+batch_size_x),:], phi_g[i:(i+batch_size_x),:], phi_h, gamma_kl, mask, self.missing)
                KL_g += self.KLD_gpi(phi_g[i:(i+batch_size_x),:], gamma_g, self.alpha_g)
            i = batch_starts_x[-1]
            mask = self.C_mask[i:,:] if self.missing else None
            ll += self.loglik_q1(self.C[i:,:], phi_g[i:,:], phi_h, gamma_kl, mask, self.missing)
            KL_g += self.KLD_gpi1(phi_g[i:,:], gamma_g, self.alpha_g)

            # compute KL_h (individual batches + last batch)
            batch_starts_y = jnp.arange(0, self.M, batch_size_y)
            for j in batch_starts_y[:-1]:
                KL_h += self.KLD_hpi(phi_h[j:(j+batch_size_y),:], gamma_h, self.alpha_h)
            j = batch_starts_y[-1]
            KL_h += self.KLD_hpi1(phi_h[j:,:], gamma_h, self.alpha_h)

            # compute KL_kl
            KL_kl = KLD_dirichlet(gamma_kl, self.alpha_pi, axis=2).sum()

        elbo = ll - KL_g - KL_h - KL_kl

        if verbose:
            print(f'ELBO: {elbo:,.3f}, Loglik: {ll:,.3f}, KL-g: {KL_g:,.3f}, KL-h: {KL_h:,.3f}, KL-kl: {KL_kl:,.3f}')

        return elbo, ll, KL_g, KL_h, KL_kl

    def loglik_fitted(self, slow=False):
        phi_g, phi_h, gamma_kl = (self.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_kl'))
        log_dir_mean, _ = log_post_dir(gamma_kl)

        post_g = jnp.argmax(phi_g, axis=1)
        post_h = jnp.argmax(phi_h, axis=1)

        if not slow:
            return utils_cnsbm.sbm_log_lik(self.C, log_dir_mean, post_g, post_h, C_mask=self.C_mask)
        else:
            return utils_cnsbm.sbm_log_lik_slow(self.C, log_dir_mean, post_g, post_h, num_cat=self.num_cat, C_mask=self.C_mask)
        
    def ICL(self, slow=False, verbose=False):
        assert self.fitted

        g_labels, h_labels = (self.posterior_dist[keys] for keys in ('g_labels', 'h_labels'))
        K_unique = len(jnp.unique(g_labels))
        L_unique = len(jnp.unique(h_labels))

        ICL_pen = ICL_penalty(self.N, self.M, K_unique, L_unique, self.num_cat)
        ll = self.loglik_fitted(slow=slow)
        
        # row/col cluster entropy
        g_entr = compute_cluster_ll(np.array(g_labels).squeeze())
        h_entr = compute_cluster_ll(np.array(h_labels).squeeze())

        ICL = ll + g_entr + h_entr - ICL_pen

        if verbose:
            print(f'ICL: {ICL:,.3f}, Loglik: {ll:,.3f}, ICL-penalty: {ICL_pen:,.3f}, g-entr: {g_entr:,.3f}, h-entr: {h_entr:,.3f}, K-eff: {K_unique}, L-eff: {L_unique}')

        self.ICL_fitted = {'ICL': ICL, 'Loglik': ll, 'ICL_pen': ICL_pen, 'g_entr': g_entr, 'h_entr': h_entr, 'K-eff': K_unique, 'L-eff': L_unique}

        return ICL

    def set_posteriors(self):
        # Obtain fitted parameters and MAP
        assert self.fitted
        
        phi_g, phi_h, gamma_kl = (self.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_kl'))
        cluster_argmax = np.array(jnp.argmax(gamma_kl, axis=2))
        dir_mean, dir_variance = post_dir(gamma_kl)

        self.posterior_dist = {'g_labels': jnp.argmax(phi_g, 1), 'h_labels': jnp.argmax(phi_h, 1), 'cluster_argmax': cluster_argmax, 'dir_mean': dir_mean, 'dir_variance': dir_variance}

        return 1
    
    def summary(self):
        """Print summary of output"""
        assert self.fitted

        _ = self.elbo(fitted=True, verbose=True)

        clusters_g = np.unique(self.posterior_dist['g_labels'])
        clusters_h = np.unique(self.posterior_dist['h_labels'])
        print(f"{len(clusters_g)} row clusters:", clusters_g)
        print(f"{len(clusters_h)} col clusters:", clusters_h)

        for threshold in [0.5, 0.75, 0.9, 0.95, 1]:
            print(f"Rows with <{threshold} probability: {int((self.fitted_params['phi_g'].max(1) < threshold).sum())}")
            print(f"Cols with <{threshold} probability: {int((self.fitted_params['phi_h'].max(1) < threshold).sum())}")

    def plt_blocks(self, plt_init=False, print_labels=False):
        assert self.fitted

        if plt_init:
            g_labels = self.phi_g.argmax(1)
            h_labels = self.phi_h.argmax(1)
            cluster_means = np.array(jnp.argmax(self.gamma_kl, axis=2))
            plt_blocks(self.C, g_labels, h_labels, cluster_means, title=' (init ' + self.rand_init + ')', print_labels=False)

        # Get posterior labels
        g_labels = self.fitted_params['phi_g'].argmax(1)
        h_labels = self.fitted_params['phi_h'].argmax(1)

        gamma_mean = self.fitted_params['gamma_kl'] / self.fitted_params['gamma_kl'].sum(axis=2, keepdims=True)
        cluster_means = np.array(jnp.argmax(gamma_mean, axis=2))

        plt_blocks(self.C, g_labels, h_labels, cluster_means, title=' (VI)', print_labels=print_labels)

    def to_cpu(self):
        """Move all JAX arrays in the class to the CPU."""
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, jax.Array):  # Check if it's a JAX array
                setattr(self, attr_name, jax.device_put(attr_value, device=jax.devices("cpu")[0]))

        gc.collect()
        return None
    
    def fitted_model_params(self):
        assert self.fitted
        params = {
            'prior': {'alpha_g': self.alpha_g, 'alpha_h': self.alpha_h, 'alpha_pi': self.alpha_pi},
            'vi_init': {'phi_g': self.phi_g, 'phi_h': self.phi_h, 'gamma_g': self.gamma_g, 'gamma_h': self.gamma_h, 'gamma_kl': self.gamma_kl},
            'vi_posterior': self.fitted_params,
            'training_history': self.training_history,
            'ICL_fitted': self.ICL_fitted
        }

        return params
    
    def export_outputs_csv(self, folderpath, model_name=None):
        foldersave = os.path.join(folderpath, model_name)
        os.makedirs(foldersave, exist_ok=True)

        post_mean = jax_array_to_csv(self.posterior_dist['dir_mean'], os.path.join(foldersave, 'sbm_fitted_pi_kl.csv'))
        cluster_argmax = pd.DataFrame(self.posterior_dist['cluster_argmax']).to_csv(os.path.join(foldersave, 'sbm_fitted_cluster_argmax.csv'), index=False)

        post_g = pd.DataFrame(self.fitted_params['phi_g']).to_csv(os.path.join(foldersave, 'sbm_fitted_phi_g.csv'), index=False)
        post_h = pd.DataFrame(self.fitted_params['phi_h']).to_csv(os.path.join(foldersave, 'sbm_fitted_phi_h.csv'), index=False)

        g_labels = pd.DataFrame(self.posterior_dist['g_labels']).to_csv(os.path.join(foldersave, 'sbm_fitted_g_labels.csv'), index=False)
        h_labels = pd.DataFrame(self.posterior_dist['h_labels']).to_csv(os.path.join(foldersave, 'sbm_fitted_h_labels.csv'), index=False)
        
    
    def save_jax_model(self, filepath):
        assert self.fitted
        params = {
            'prior': {'alpha_g': self.alpha_g, 'alpha_h': self.alpha_h, 'alpha_pi': self.alpha_pi},
            'vi_init': {'phi_g': self.phi_g, 'phi_h': self.phi_h, 'gamma_g': self.gamma_g, 'gamma_h': self.gamma_h, 'gamma_kl': self.gamma_kl},
            'vi_posterior': self.fitted_params,
            'training_history': self.training_history,
            'posterior_dist': self.posterior_dist,
            'ICL_fitted': self.ICL_fitted
        }
        
        # Save as pickle
        print("Saving model...")
        with open(filepath, "wb") as f:
            pickle.dump(params, f)
    
    def load_jax_model(self, filepath):
        print("Loading saved model...")
        with open(filepath, "rb") as f:
            params= pickle.load(f)
        
        self.alpha_g, self.alpha_h, self.alpha_pi = (params['prior'][key] for key in ['alpha_g', 'alpha_h', 'alpha_pi'])
        self.phi_g, self.phi_h, self.gamma_g, self.gamma_h, self.gamma_kl = (params['vi_init'][key] for key in ['phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl'])
        self.fitted_params = params['vi_posterior']
        self.training_history = params['training_history']
        self.ICL_fitted = params['ICL_fitted']

        assert self.phi_g.shape[1] == self.K and self.phi_h.shape[1] == self.L
        assert self.gamma_g.shape[0] == self.K and self.gamma_h.shape[0] == self.L
        assert self.gamma_kl.shape == (self.K, self.L, self.num_cat)

        self.fitted = True
        self.set_posteriors()

    def load_jax_model_dict(self, params):
        print("Loading saved model...")
        
        self.alpha_g, self.alpha_h, self.alpha_pi = (params['prior'][key] for key in ['alpha_g', 'alpha_h', 'alpha_pi'])
        self.phi_g, self.phi_h, self.gamma_g, self.gamma_h, self.gamma_kl = (params['vi_init'][key] for key in ['phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl'])
        self.fitted_params = params['vi_posterior']
        self.training_history = params['training_history']
        self.ICL_fitted = params['ICL_fitted']

        assert self.phi_g.shape[1] == self.K and self.phi_h.shape[1] == self.L
        assert self.gamma_g.shape[0] == self.K and self.gamma_h.shape[0] == self.L
        assert self.gamma_kl.shape == (self.K, self.L, self.num_cat)

        self.fitted = True
        self.set_posteriors()



