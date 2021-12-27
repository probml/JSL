# JSL: JAX State-Space models (SSM) Library

<img width="500" alt="image" src="https://user-images.githubusercontent.com/4108759/146819263-7d476231-22c9-4e03-98c6-a6b300d99c5e.png">

JSL is a JAX library for Bayesian inference in linear and non-linear Gaussian state-space models.
We assume that the model parameters are known, and just focus on state estimation.
For linear dynamical systems (LDS), we support Kalman filtering and RTS smoothing.
For nonlinear dynamical systems (NLDS), we support extended Kalman filtering (EKF) with full and diagonal covariance,
unscented Kalman filtering (UKF) and bootstrap particle filtering (PF).

# Installation

```
pip install git+git://github.com/probml/jsl
```

# Examples

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

# Authors
  
Gerardo Durán-Martín ([@gerdm](https://github.com/gerdm)), Aleyna Kara([@karalleyna](https://github.com/karalleyna)), Kevin Murphy ([@murphyk](https://github.com/murphyk)).  
