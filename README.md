# Copy Number Stochastic Block Model (CNSBM) code for MLCB 2025

We recommend using a conda environment. Otherwise you can just install the packages in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate cnsbm
```

Compatibility note: in addition to exporting outputs via csv, we can save models as .pickle. There is a numpy 2.0 vs numpy <2.0 incompatibility when loading these pickles.

# Simple usage

```python
import os
import jax.numpy as jnp
from model import CNSBM

cwd = os.getcwd()

# convert categorical matrix to jax array
# missing values are encoded as -1. Categories need to be encoded starting from 0.
# The number of categories will be inferred by C.max()
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