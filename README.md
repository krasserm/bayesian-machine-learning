## Bayesian machine learning notebooks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4050922.svg)](https://doi.org/10.5281/zenodo.4050922)

This repository is a collection of notebooks about *Bayesian Machine Learning*. The following links display 
some of the notebooks via [nbviewer](https://nbviewer.jupyter.org/) to ensure a proper rendering of formulas.

- [Bayesian regression with linear basis function models](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-linear-regression/bayesian_linear_regression.ipynb). 
  Introduction to Bayesian linear regression. Implementation from scratch with plain NumPy as well as usage of scikit-learn 
  for comparison. See also 
  [PyMC4 implementation](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-linear-regression/bayesian_linear_regression_pymc4.ipynb) and 
  [PyMC3 implementation](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-linear-regression/bayesian_linear_regression_pymc3.ipynb).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/bayesian-machine-learning/blob/dev/gaussian-processes/gaussian_processes.ipynb)
  [Gaussian processes](https://krasserm.github.io/2018/03/19/gaussian-processes/). 
  Introduction to Gaussian processes for regression. Example implementations with plain NumPy/SciPy as well as with libraries 
  scikit-learn and GPy ([requirements.txt](gaussian-processes/requirements.txt)). 

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/bayesian-machine-learning/blob/dev/gaussian-processes/gaussian_processes_classification.ipynb)
  [Gaussian processes for classification](https://krasserm.github.io/2020/11/04/gaussian-processes-classification/). 
  Introduction to Gaussian processes for classification. Example implementations with plain NumPy/SciPy as well as with 
  scikit-learn ([requirements.txt](gaussian-processes/requirements.txt)). 

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-optimization/bayesian_optimization.ipynb)
  [Bayesian optimization](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-optimization/bayesian_optimization.ipynb). 
  Introduction to Bayesian optimization. Example implementations with plain NumPy/SciPy as well as with libraries 
  scikit-optimize and GPyOpt. Hyper-parameter tuning as application example.  

- [Variational inference in Bayesian neural networks](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-neural-networks/bayesian_neural_networks.ipynb). 
  Demonstrates how to implement a Bayesian neural network and variational inference of network parameters. Example implementation 
  with Keras ([requirements.txt](bayesian-neural-networks/requirements.txt)). See also 
  [PyMC4 implementation](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/bayesian-neural-networks/bayesian_neural_networks_pymc4.ipynb).

- [Reliable uncertainty estimates for neural network predictions](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/noise-contrastive-priors/ncp.ipynb). 
  Uses noise contrastive priors in Bayesian neural networks to get more reliable uncertainty estimates for OOD data.
  Implemented with Tensorflow 2 and Tensorflow Probability ([requirements.txt](noise-contrastive-priors/requirements.txt)).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/bayesian-machine-learning/blob/dev/latent-variable-models/latent_variable_models_part_1.ipynb)
  [Latent variable models, part 1: Gaussian mixture models and the EM algorithm](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/latent-variable-models/latent_variable_models_part_1.ipynb).
  Introduction to the expectation maximization (EM) algorithm and its application to Gaussian mixture models. Example
  implementation with plain NumPy/SciPy and scikit-learn for comparison. See also 
  [PyMC3 implementation](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/latent-variable-models/latent_variable_models_part_1_pymc3.ipynb).

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krasserm/bayesian-machine-learning/blob/dev/latent-variable-models/latent_variable_models_part_2.ipynb)
  [Latent variable models, part 2: Stochastic variational inference and variational autoencoders](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/latent-variable-models/latent_variable_models_part_2.ipynb). 
  Introduction to stochastic variational inference with variational autoencoder as application example. Implementation 
  with Tensorflow 2.x.

- [Deep feature consistent variational autoencoder](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/autoencoder-applications/variational_autoencoder_dfc.ipynb). 
  Describes how a perceptual loss can improve the quality of images generated by a variational autoencoder. Example 
  implementation with Keras.  

- [Conditional generation via Bayesian optimization in latent space](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/dev/autoencoder-applications/variational_autoencoder_opt.ipynb). 
  Describes an approach for conditionally generating outputs with desired properties by doing Bayesian optimization in 
  latent space learned by a variational autoencoder. Example application implemented with Keras and GPyOpt.
