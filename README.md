The code uses minimal common packages (numpy, jax, scipy, matplotlib, torch, etc.) alongside the XGBoost package, and should be clear from the code files. The argparse for each script is set to default values, and passing different values allows runing the hyperparameters sweeps shown in the paper. In the code it is also necessary to manually define the result_dir path (an argument to pass is not provided).

We release our code under the MIT license.
