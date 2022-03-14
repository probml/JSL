import jax.numpy as jnp
from jax import random
from jsl.nlds.base import NLDS

from jsl.seql.agents.eekf_agent import eekf
from jsl.seql.environments.base import make_random_linear_classification_environment
from jsl.seql.utils import train

def fz(x): return x
def fx(w, x): 
    return (x @ w)[None, ...]
def Rt(w, x): return (x @ w * (1 - x @ w))[None, None]

def callback_fn(**kwargs):
    y_pred, _ = kwargs["preds"]
    y_test = jnp.squeeze(kwargs["Y_test"])
    print("Accuracy: ", jnp.mean(jnp.squeeze(y_pred)==y_test))

def main():
    key = random.PRNGKey(0)
    ntrain = 200  # 80% of the data
    ntest = 50  # 20% of the data
    nfeatures, nclasses = 2, 2

    env = make_random_linear_classification_environment(key,
                                                  nfeatures,
                                                  ntrain,
                                                  ntest,
                                                  ntargets=nclasses)

    obs_noise = 0.01
    Pt = jnp.eye(nfeatures) * 0.0
    P0 = jnp.eye(nfeatures) * 2.0
    mu0 = jnp.zeros((nfeatures,))

    nlds = NLDS(fz, fx, Pt, Rt, mu0, P0)
    kf_agent = eekf(nlds, obs_noise=obs_noise)

    mu0 = jnp.zeros((nfeatures,))
    Sigma0 = jnp.eye(nfeatures)

    belief = kf_agent.init_state(mu0, Sigma0)

    nsteps = 100
    _, unused_rewards = train(belief, kf_agent, env,
                                nsteps=nsteps,
                                callback=callback_fn)
                 

    

if __name__ == "__main__":
    main()
