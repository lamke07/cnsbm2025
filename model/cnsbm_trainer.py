from .utils_trainer import (
    row_col_redundant, select_redundant_clusters, clusters_to_split_single,
    new_cluster_labels_single, get_cluster_sizes, new_cluster_labels_block,
    row_col_impurity, reset_min_clusters, reset_duplicated_clusters
)
from .cnsbm import CNSBM
import pickle
import pandas as pd # remove this later
import numpy as np
import jax.numpy as jnp

class CNSBMTrainer:
    def __init__(self, init_sbm):
        # assign first sbm model
        assert init_sbm.fitted
        self.filled_sbm = init_sbm

        # Matrix Size (C_fill for cluster expansion, C for actual data fitting with missing values)
        self.C_fill = self.filled_sbm.C
        if self.filled_sbm.missing:
            self.C = jnp.where(self.filled_sbm.C_mask, self.filled_sbm.C, -1)
            self.fill_na = self.filled_sbm.fill_na
        else:
            self.C = self.C_fill
            self.fill_na = 0
        self.seed = self.filled_sbm.seed
        self.propensity_mode = self.filled_sbm.propensity_mode
        self.N, self.M = self.C.shape
        self.num_cat = int(self.C.max() + 1)

        # Cluster dimensions
        self.K, self.L = self.filled_sbm.K, self.filled_sbm.L
        self.num_cat = self.filled_sbm.num_cat
        
        # need to check for fitted SBM first
        # Model history and model params
        fitted_model_params = init_sbm.fitted_model_params().copy()
        self.model_history = [fitted_model_params['training_history'][-1]]
        self.model_history[0].update(fitted_model_params['ICL_fitted'])
        self.model_history[0].update({'model_version': -1, 'strategy': 'base'})
        self.model_history_params = {-1: fitted_model_params}

        self.curr_iter = 0

    def fill_clusters(self,
                      max_iter=20, max_switch=6,
                      empty_threshold=1e-6, entr_threshold=0.2, split_threshold=None, min_split_threshold=0.05,
                      concentration=0.7,
                      batch_vi=True, stochastic_vi_params=None,
                      multi_split=True, verbose=True, max_vi_iter=100, vi_batch_print=1, fill_strategy='spectral', alt_fill_strategy='spectral',
                      plt_blocks=False
                      ):
        # Whether to use prefilled clusters from an earlier iteration
        curr_elbo, _, _, _, _ = self.filled_sbm.elbo(fitted=True, verbose=verbose)
        sbm_new = sbm_prop = self.filled_sbm
        if not batch_vi:
            assert stochastic_vi_params is not None
            batch_size, local_max_iters, local_tol, eta_0, tau, kappa, batches_per_epoch, tol = (stochastic_vi_params[key] for key in ['batch_size', 'local_max_iters', 'local_tol', 'eta_0', 'tau', 'kappa', 'batches_per_epoch', 'tol'])

        curr_switch, curr_priority = 0, 'row'
        max_iter = self.curr_iter + max_iter

        while self.curr_iter <= max_iter and curr_switch <= max_switch:
            print(f"\nIteration {self.curr_iter}\n")

            # Obtain fitted parameters and MAP
            phi_g, phi_h, _ = (sbm_new.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_kl')) # 'gamma_g', 'gamma_h'
            g_labels, h_labels, dir_mean, _ = (sbm_new.posterior_dist[keys] for keys in ('g_labels', 'h_labels', 'dir_mean', 'dir_variance'))
            
            # 1. Check for row and column impurities, # 2. Obtain duplicated or empty clusters, new cluster proposals, and new labels, # 3. Out of the empty/duplicated clusters, select which one to split along to
            row_impurities, col_impurities, dir_mean_entr_masked = row_col_impurity(dir_mean, empty_threshold=empty_threshold, entr_threshold=entr_threshold)
            g_empty, g_duplicated, h_empty, h_duplicated = row_col_redundant(phi_g, phi_h, dir_mean, empty_threshold=empty_threshold, posterior_map=False)
            split_direction, redundant_clusters = select_redundant_clusters(g_empty, g_duplicated, h_empty, h_duplicated, priority=curr_priority)
                        
            if split_direction is None or (len(row_impurities) == 0 and len(col_impurities) == 0):
                print("No impurities or no more clusters to split - analyze output or set lower entr_threshold")
                break

            # get new cluster labels
            if len(redundant_clusters) == 0 or fill_strategy != 'spectral_bi':
                # when only one row or one col is available, spectral_bi cannot choose a block
                curr_fill_strat = alt_fill_strategy if fill_strategy == 'spectral_bi' else fill_strategy
                cluster_candidates = clusters_to_split_single(
                    dir_mean, split_direction, empty_threshold=empty_threshold, verbose=verbose, min_split_threshold=min_split_threshold
                )
                g_labels_new, h_labels_new = new_cluster_labels_single(
                    self.C_fill, dir_mean, g_labels, h_labels, cluster_candidates, split_direction, redundant_clusters,
                    how=curr_fill_strat, split_threshold=split_threshold, verbose=verbose, multi_split=multi_split
                )
            elif fill_strategy == 'spectral_bi':
                cluster_sizes, sorted_indices, sorted_sizes = get_cluster_sizes(g_labels, h_labels, self.K, self.L)
                cluster_candidates = [x for x in sorted_indices if dir_mean_entr_masked[x[0], x[1]] > entr_threshold] # add size threshold?
                
                g_labels_new, h_labels_new = new_cluster_labels_block(
                    self.C_fill, g_labels, h_labels,
                    cluster_candidates, redundant_clusters,
                    how=fill_strategy, verbose=verbose, multi_split=True
                )

            # Fit new model with new initializations and warm starts for gamma
            print("\nFitting Model ...")
            sbm_prop = CNSBM(self.C, self.K, self.L, init_clusters={'g': g_labels_new, 'h': h_labels_new}, rand_init='init', concentration=concentration, warm_start=True, fill_na=self.fill_na, seed=self.seed+self.curr_iter, propensity_mode=self.propensity_mode)
            sbm_prop.update_gammas_warm(update_ind=False)
            if batch_vi:
                _ = sbm_prop.batch_vi(max_vi_iter, batch_print=vi_batch_print)
            else:
                _ = sbm_prop.stochastic_vi(num_iters=max_vi_iter, batch_size=batch_size, batch_print=vi_batch_print, local_max_iters=local_max_iters, local_tol=local_tol, eta_0=eta_0, tau=tau, kappa=kappa, batches_per_epoch=batches_per_epoch, tol=tol)

            # Analytics and model update
            new_elbo, _, _, _, _ = sbm_prop.elbo(fitted=True, verbose=verbose)
            sbm_prop.ICL(verbose=verbose)
            self.update_model_history(sbm_prop, self.curr_iter, strategy=fill_strategy)

            # Check which direction to split in next
            print(f"Old ELBO: {curr_elbo:,.3f} | New ELBO: {new_elbo:,.3f}")
            
            if new_elbo > curr_elbo:
                print("ELBO improved, keeping new model and current priority...")
                curr_elbo = new_elbo
                sbm_new = sbm_prop
                curr_switch = 0
            else:
                curr_priority = 'col' if curr_priority == 'row' else 'row'
                curr_switch += 1
                print(f"ELBO not improved, keeping old model | nr switches: {curr_switch}")
                if curr_switch > max_switch:
                  print(f"Max number of switches reached. Consider changing `fill_strategy` or `concentration` parameters.")

            # plot output of proposed model
            if plt_blocks:
                sbm_prop.plt_blocks(plt_init=False)
            self.curr_iter += 1

        # Final values
        self.filled_sbm = sbm_new

    def select_model_version(self, model_version=None, select_best=False, verbose=False, criterion='ICL'):
        if select_best:
            model_version = pd.DataFrame(self.model_history).sort_values(criterion).iloc[-1,:]['model_version']
        else:
            assert model_version is not None

        if verbose:
            print(f"Setting to model version {model_version} based on {criterion}")
        params = self.model_history_params[model_version]
            
        self.filled_sbm.alpha_g, self.filled_sbm.alpha_h, self.filled_sbm.alpha_pi = (params['prior'][key] for key in ['alpha_g', 'alpha_h', 'alpha_pi'])
        self.filled_sbm.phi_g, self.filled_sbm.phi_h, self.filled_sbm.gamma_g, self.filled_sbm.gamma_h, self.filled_sbm.gamma_kl = (params['vi_init'][key] for key in ['phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl'])
        self.filled_sbm.fitted_params = params['vi_posterior']
        self.filled_sbm.training_history = params['training_history']
        self.filled_sbm.ICL_fitted = params['ICL_fitted']

        self.filled_sbm.set_posteriors()
            
    def update_model_history(self, sbm_model, curr_iter, strategy=''):
        fitted_model_params = sbm_model.fitted_model_params().copy()
        self.model_history.append(fitted_model_params['training_history'][-1])
        self.model_history[-1].update(fitted_model_params['ICL_fitted'])
        self.model_history[-1].update({'model_version': curr_iter, 'strategy': strategy})
        self.model_history_params.update({curr_iter: fitted_model_params})

    def reset_min_clusters(self, min_samples_g=10, min_samples_h=10, 
                           max_iter=20, max_patience=10, concentration=0.8, max_vi_iter=100, vi_batch_print=1, batch_vi=True, stochastic_vi_params=None, 
                           direction='multi', verbose=True, criterion='ICL'):

        if criterion == 'ICL':
            curr_max = self.filled_sbm.ICL()
        else:
            curr_max, _, _, _, _ = self.filled_sbm.elbo(fitted=True, verbose=False)
        max_iter, curr_patience = self.curr_iter + max_iter, 0

        while self.curr_iter <= max_iter and curr_patience <= max_patience:
            g_labels, h_labels = (self.filled_sbm.posterior_dist[keys] for keys in ('g_labels', 'h_labels'))
            alpha_pi = self.filled_sbm.alpha_pi
            gamma_g, gamma_h, gamma_kl = (self.filled_sbm.fitted_params[keys] for keys in ('gamma_g', 'gamma_h', 'gamma_kl'))

            curr_direction = np.random.choice(['row', 'col']) if direction != 'multi' else direction
            gamma_kl = reset_min_clusters(g_labels.tolist(), h_labels.tolist(), gamma_kl, alpha_pi, min_samples_g=min_samples_g, min_samples_h=min_samples_h, direction=curr_direction)

            sbm_prop = CNSBM(self.C, self.K, self.L, 
                    init_clusters={'g': g_labels, 'h': h_labels}, rand_init='init',
                    gammas={'gamma_g': gamma_g, 'gamma_h': gamma_h, 'gamma_kl': gamma_kl}, 
                    concentration=concentration, warm_start=True, fill_na=self.fill_na,
                    seed=self.seed + self.curr_iter, propensity_mode=self.propensity_mode
                )

            if batch_vi:
                _ = sbm_prop.batch_vi(max_vi_iter, batch_print=vi_batch_print)
            else:
                assert stochastic_vi_params is not None
                batch_size, local_max_iters, local_tol, eta_0, tau, kappa, batches_per_epoch, tol = (stochastic_vi_params[key] for key in ['batch_size', 'local_max_iters', 'local_tol', 'eta_0', 'tau', 'kappa', 'batches_per_epoch', 'tol'])
                _ = sbm_prop.stochastic_vi(num_iters=max_vi_iter, batch_size=batch_size, batch_print=vi_batch_print, local_max_iters=local_max_iters, local_tol=local_tol, eta_0=eta_0, tau=tau, kappa=kappa, batches_per_epoch=batches_per_epoch, tol=tol)

            sbm_prop.ICL(verbose=True)
            if criterion == 'ICL':
                new_max = sbm_prop.ICL_fitted['ICL']
            else:
                new_max, _, _, _, _ = sbm_prop.elbo(fitted=True, verbose=verbose)

            self.update_model_history(sbm_prop, self.curr_iter, strategy='reset_min')
            print(f"Old {criterion}: {curr_max:,.3f} | New {criterion}: {new_max:,.3f}")

            if new_max > curr_max:
                print(f"{criterion} improved, keeping new model and current priority...\n")
                curr_max = new_max
                curr_patience=0
                self.filled_sbm = sbm_prop
            else:
                curr_patience+=1
                print(f"{criterion} not improved, keeping old model\n")

            self.curr_iter += 1

    def reset_duplicated_clusters(self, empty_threshold=1e-6, posterior_map=True,
                                  max_iter=10, max_patience=10, concentration=0.8, max_vi_iter=100, vi_batch_print=1, batch_vi=True, stochastic_vi_params=None,
                                  direction='multi', verbose=True, criterion='ICL'):
        if criterion == 'ICL':
            curr_max = self.filled_sbm.ICL()
        else:
            curr_max, _, _, _, _ = self.filled_sbm.elbo(fitted=True, verbose=False)
        max_iter, curr_patience = self.curr_iter + max_iter, 0

        while self.curr_iter <= max_iter and curr_patience <= max_patience:
            g_labels, h_labels, dir_mean = (self.filled_sbm.posterior_dist[keys] for keys in ('g_labels', 'h_labels', 'dir_mean'))
            alpha_pi = self.filled_sbm.alpha_pi
            phi_g, phi_h, gamma_g, gamma_h, gamma_kl = (self.filled_sbm.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl'))
            
            curr_direction = np.random.choice(['row', 'col']) if direction != 'multi' else direction
            gamma_kl = reset_duplicated_clusters(phi_g, phi_h, dir_mean, gamma_kl, alpha_pi, empty_threshold=empty_threshold, verbose=verbose, posterior_map=posterior_map, direction=curr_direction)

            sbm_prop = CNSBM(self.C, self.K, self.L, 
                    init_clusters={'g': g_labels, 'h': h_labels}, rand_init='init',
                    gammas={'gamma_g': gamma_g, 'gamma_h': gamma_h, 'gamma_kl': gamma_kl}, 
                    concentration=concentration, warm_start=True, fill_na=self.fill_na,
                    seed=self.seed + self.curr_iter, propensity_mode=self.propensity_mode
                )

            if batch_vi:
                _ = sbm_prop.batch_vi(max_vi_iter, batch_print=vi_batch_print)
            else:
                assert stochastic_vi_params is not None
                batch_size, local_max_iters, local_tol, eta_0, tau, kappa, batches_per_epoch, tol = (stochastic_vi_params[key] for key in ['batch_size', 'local_max_iters', 'local_tol', 'eta_0', 'tau', 'kappa', 'batches_per_epoch', 'tol'])
                _ = sbm_prop.stochastic_vi(num_iters=max_vi_iter, batch_size=batch_size, batch_print=vi_batch_print, local_max_iters=local_max_iters, local_tol=local_tol, eta_0=eta_0, tau=tau, kappa=kappa, batches_per_epoch=batches_per_epoch, tol=tol)
            
            new_max, _, _, _, _ = sbm_prop.elbo(fitted=True, verbose=verbose)
            sbm_prop.ICL(verbose=True)
            if criterion == 'ICL':
                new_max = sbm_prop.ICL_fitted['ICL']
            else:
                new_max, _, _, _, _ = sbm_prop.elbo(fitted=True, verbose=verbose)

            self.update_model_history(sbm_prop, self.curr_iter, strategy='reset_duplicates')
            print(f"Old {criterion}: {curr_max:,.3f} | New {criterion}: {new_max:,.3f}")

            if new_max > curr_max:
                print(f"{criterion} improved, keeping new model...\n")
                curr_max = new_max
                curr_patience=0
                self.filled_sbm = sbm_prop
            else:
                curr_patience+=1
                print(f"{criterion} not improved, keeping old model\n")

            self.curr_iter += 1

    def clear_model_cache(self):
        df = pd.DataFrame(self.model_history)

        for y in ['ELBO', 'ICL']:
            df[y] = np.array([float(x) for x in df[y]], dtype=np.float32)

        df = df.sort_values(['ICL', 'ELBO'])
        df1 = df.drop_duplicates(subset=['K-eff', 'L-eff', 'strategy'], keep='last')

        self.model_history = [v for v in self.model_history if v['model_version'] in list(df1['model_version'])]
        self.model_history_params = {k: self.model_history_params[k] for k in list(df1['model_version'])}

    def save_trainer_model(self, filepath):
        params = {
            'model_history': self.model_history,
            'model_history_params': self.model_history_params,
        }
        
        # Save as pickle
        print(f"Saving model... to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(params, f)

        # with open(os.path.join(cwd, 'trainer_test.pkl'), "rb") as f:
        #     params1 = pickle.load(f)