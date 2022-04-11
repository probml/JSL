import typing_extensions

import chex
from typing import NamedTuple, Tuple, Callable

BeliefState = NamedTuple
Info = NamedTuple
AgentInitFn = Callable
SampleFn = Callable


class AgentUpdateFn(typing_extensions.Protocol):

    def __call__(self,
                 belief: BeliefState,
                 x: chex.Array,
                 y: chex.Array) -> Tuple[BeliefState, Info]:
        '''
        It updates the belief given training data x and y.
        '''
        ...


class AgentPredictFn(typing_extensions.Protocol):

    def __call__(self,
                 belief: BeliefState,
                 x: chex.Array) -> chex.Array:
        '''
        It predicts the outputs of x using the current belief state.
        '''
        ...


class SamplePredictiveFn(typing_extensions.Protocol):

    def __call__(self,
                 key: chex.PRNGKey,
                 belief: BeliefState,
                 x: chex.Array,
                 nsamples: int
                 ) -> chex.Array:
        ...


class Agent(NamedTuple):
    '''
    Agent interface.
    '''
    init_state: AgentInitFn
    update: AgentUpdateFn
    predict: AgentPredictFn
    sample_predictive: SamplePredictiveFn
