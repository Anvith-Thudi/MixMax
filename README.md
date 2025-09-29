#MixMax

This repository contains the code used to run the experiments in "MixMax: Distributional Robustness in Function Space via Optimal Data Mixtures" published at ICLR 2025


The environment used to run the experiments is specified in the requirements.txt, but this also contains some unecessary packages. See the imports for the relevant experiments for the specific packages used (e.g., numpy, jax, xgboost, etc.).

The argparse for each script is set to default values, and passing different values allows runing the hyperparameters sweeps shown in the paper. In the code it is also necessary to manually define the result_dir path (an argument to pass is not provided).

We release our code under the MIT license.

