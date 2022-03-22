# JSL: JAX State-Space models (SSM) Library

<p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/4108759/146819263-7d476231-22c9-4e03-98c6-a6b300d99c5e.png">
</p>

JSL is a JAX library for Bayesian inference in state space models.
We support discrete state spaces (HMM) and continuous state spaces.
For HMMs, we support exact inference (forwards-backwards and Viterbi) and MLE learning (using EM or SGD).
For continuous SSMs, we assume that the model parameters are known, and just focus on state estimation.
We support linear and non-linear dynamics/ observations; we assume all noise is Gaussian.
For linear dynamical systems (LDS), we support Kalman filtering and RTS smoothing.
For nonlinear dynamical systems (NLDS), we support extended Kalman filtering (EKF) with full and diagonal covariance,
unscented Kalman filtering (UKF) and  particle filtering (PF).

# Installation

We assume you have already installed [JAX](https://github.com/google/jax#installation) and
[Tensorflow](https://www.tensorflow.org/install),
since the details on how to do this depend on whether you have a CPU, GPU, etc.
(This step is not necessary in Colab.)

Now install these packages:

```
!pip install --upgrade git+https://github.com/google/flax.git
!pip install --upgrade tensorflow-probability
!pip install git+git://github.com/blackjax-devs/blackjax.git
!pip install git+git://github.com/deepmind/distrax.git
!pip install superimport 
!pip install fire
```

Then install JSL:
```
!pip install git+git://github.com/probml/jsl
```
Alternatively, you can clone the repo locally, into say `~/github/JSL`, and then install it as a package, as follows:
```
!git clone https://github.com/probml/JSL.git
cd JSL
!pip install -e .
```

# Running the demos

You can see how to use the library by looking at some of the demos.
You can run the demos from inside a notebook like this
```
%run JSL/jsl/demos/kf_tracking.py
%run JSL/jsl/demos/hmm_casino_em_train.py
```

Or from inside an ipython shell like this
```
from jsl.demos import kf_tracking
figdict = kf_tracking.main()
```

Most of the demos create figures. If you want to save them (in both png and pdf format),
you need to specify the FIGDIR environment variable, like this:
```
import os
os.environ["FIGDIR"]='/Users/kpmurphy/figures'

from jsl.demos.plot_utils import savefig
savefig(figdict)
```

# Authors
  
Gerardo Durán-Martín ([@gerdm](https://github.com/gerdm)), Aleyna Kara([@karalleyna](https://github.com/karalleyna)), Kevin Murphy ([@murphyk](https://github.com/murphyk)).  
