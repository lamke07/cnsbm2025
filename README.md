# CN-SBM: Categorical Block Modelling For Primary and Residual Copy Number Variation
This repository contains the implementation of the model described in the paper: **CN-SBM: Categorical Block Modelling For Primary and Residual Copy Number Variation** ([arXiv:2506.22963](https://arxiv.org/abs/2506.22963)), to appear in MLCB 2025.

# Note on NumPy Version
If you plan to use pickle files in an environment with NumPy < 2, you should install numpy < 2 for compatibility. Otherwise, it is fine to use NumPy 2 or newer.

# Copy Number Stochastic Block Model (CNSBM) code

We recommend using a conda environment. Otherwise you can just install the packages in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate cnsbm
```

# Simple usage

```python
import os
import jax.numpy as jnp
from model import CNSBM

cwd = os.getcwd()

# C is a categorical matrix (integer-encoded categories starting from 0),
# missing values are encoded as -1. The number of categories will be inferred by C.max().
# For an example of how to construct and use C, see cn_vi-simple.ipynb in this repository
C = jnp.asarray(C)
K, L = 15, 10

# Initialize Jax model
sbm_test = CNSBM(C, K, L, rand_init='spectral_bi', fill_na=2)
# Run batch variational inference
_ = sbm_test.batch_vi(75, batch_print=1, fitted=False, tol=1e-6)

# plot reordered output and get summary information
sbm_test.plt_blocks(plt_init=True)
sbm_test.summary()
_ = sbm_test.ICL(verbose=True, slow=True)

# Save model outputs and export cluster labels / probabilities
os.makedirs(os.path.join(cwd, 'output'), exist_ok=True)
sbm_test.export_outputs_csv(os.path.join(cwd, 'output'), model_name='test_sbm')
sbm_test.save_jax_model(os.path.join(cwd, 'output', f'test_sbm.pickle'))
```