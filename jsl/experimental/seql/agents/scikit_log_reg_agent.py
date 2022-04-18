'''# UNUSED

import jax.numpy as jnp

from sklearn.linear_model import LogisticRegression

import chex
import warnings
from typing import Any, NamedTuple

from jsl.experimental.seql.agents.agent_utils import Memory
from jsl.experimental.seql.agents.base import Agent

Params = Any


class BeliefState(NamedTuple):
    params: Params = None,
    clf: Params = None


class Info(NamedTuple):
    loss: float


def scikit_log_reg_agent(penalty='l2',
                         dual: bool = False,
                         tol: float = 1e-4,
                         C: float = 1.0,
                         fit_intercept: bool = True,
                         intercept_scaling: float = 1.,
                         class_weight=None,
                         random_state: int = None,
                         solver='lbfgs',
                         max_iter: int = 100,
                         multi_class='auto',
                         verbose: int = 0,
                         warm_start: bool = False,
                         n_jobs: int = None,
                         l1_ratio: float = None,
                         obs_noise=0.1,
                         threshold=1,
                         buffer_size=0):

    classification = True
    memory = Memory(buffer_size)

    def init_state(*params: Params):
        logreg = LogisticRegression(penalty=penalty,
                                    dual=dual,
                                    tol=tol,
                                    C=C,
                                    fit_intercept=fit_intercept,
                                    intercept_scaling=intercept_scaling,
                                    class_weight=class_weight,
                                    random_state=random_state,
                                    solver=solver,
                                    max_iter=max_iter,
                                    multi_class=multi_class,
                                    verbose=verbose,
                                    warm_start=warm_start,
                                    n_jobs=n_jobs,
                                    l1_ratio=l1_ratio)
        return BeliefState(params=params, clf=logreg)

    def update(key: chex.PRNGKey,
               belief: BeliefState,
               x: chex.Array,
               y: chex.Array):
        assert buffer_size >= len(x)
        x_, y_ = memory.update(x, y)

        if len(x_) < threshold:
            warnings.warn("There should be more data.", UserWarning)
            return belief, None

        classifier = belief.clf
        logreg = classifier.fit(x_, jnp.squeeze(y_))
        loss = logreg.score(x_, jnp.squeeze(y_))
        return BeliefState(logreg.get_params, logreg), Info(loss)

    def _apply(params: chex.ArrayTree,
              x: chex.Array):
        return belief.clf.predict_log_proba(x)

    def sample_params(key: chex.PRNGKey,
                      belief: BeliefState):
        return belief.params

    return Agent(classification, init_state, update, _apply, sample_params)
'''