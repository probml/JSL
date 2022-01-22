# JSL: JAX State-Space models (SSM) Library

<p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/4108759/146819263-7d476231-22c9-4e03-98c6-a6b300d99c5e.png">
</p>

JSL is a JAX library for Bayesian inference in linear and non-linear Gaussian state-space models.
We assume that the model parameters are known, and just focus on state estimation.
For linear dynamical systems (LDS), we support Kalman filtering and RTS smoothing.
For nonlinear dynamical systems (NLDS), we support extended Kalman filtering (EKF) with full and diagonal covariance,
unscented Kalman filtering (UKF) and bootstrap particle filtering (PF).

# Installation

We assume you have already installed [JAX](https://github.com/google/jax#installation) and
[Tensorflow](https://www.tensorflow.org/install),
since the details on how to do this depend on whether you have a CPU, GPU, etc.
(This step is not necessary in Colab.)

Now install these packages:

```
pip install --upgrade git+https://github.com/google/flax.git
pip install blackjax
pip install superimport 
pip install fire
```

Then install JSL:
```
pip install git+git://github.com/probml/jsl
```
Alternatively, you can clone the repo locally, into say `~/github/JSL`, and then install it as a package, as follows:
```
git clone https://github.com/probml/JSL.git
cd JSL
pip install -e .
```

# Examples

To run the examples included in JSL from the command line, clone the repository and `cd` into `JSL`. Then run

```
python -m jsl run_demo [NAME_OF_DEMO]
```

To see the available demos, run

```
python -m jsl list_demos
```

Alternatively, you can run any demo directly from JSL after instalation by importing the desired demo and running its `main()` function as follows:

```python
>>> from jsl.demos import NAME_OF_DEMO
>> figures = NAME_OF_DEMO.main()
```

The resulting variable `figures` is a dictionary with values the output figures and keys the recommended figure names.


## Current available demos

**Basic examples of a Kalman Filter. Based on the idea of tracking missiles**  
Script: `kf_tracking_demo` Single example  
Script: `kf_parallel_demo` In parallel

**continous-time Kalman filtering of a linear dynamical system with imaginary eigenvalues**  
Script: `kf_continuous_circle_demo`

**Kalman filtering of a linear dynamical system with imaginary eigenvalues**  
Script: `kf_continuous_circle_demo`

**Filtering a non-linear system with Gaussian noise using the bootstrap filter**  
Script: `bootstrap_filter_demo`

**Estimating a non-linear dynamical system using EKF and UKF**  
Script: `ekf_vs_ukf.py`

**Estimating a continuous-time non-linear dynamical system using EKF**  
Script: `ekf_continuous_demo.py`

**Sequentially estimating the parameters of a linear regresison model using the Kalman filter (KF) algorithm**  
Script: `linreg_kf_demo.py`

**Sequentially estimating the parameters of a logistic regression model using the exponential-family EKF (EEKF)**  
Script: `eekf_logistic_regression_demo.py`

**Sequentially learning a multi-layered perceptron on 1d nonlinear regression problem using EKF and UKF**  
Script: `ekf_vs_ukf_mlp_demo.py`, `ekf_mlp_anim_demo.py`.
The animation script produces <a href="https://github.com/probml/probml-data/blob/main/data/ekf_mlp_demo.mp4">this video</a>.

**Comparing EKF v.s. bootstrap filtering when there is non-gaussian noise added to the system**  
Script `pendulum_1d_demo`

# Authors
  
Gerardo Durán-Martín ([@gerdm](https://github.com/gerdm)), Aleyna Kara([@karalleyna](https://github.com/karalleyna)), Kevin Murphy ([@murphyk](https://github.com/murphyk)).  
