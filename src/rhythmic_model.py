from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.types.state import StateVector, StateVectors
from stonesoup.types.array import CovarianceMatrix
from stonesoup.base import Property
import numpy as np
from typing import Type

from rhythmic_sharing import RhythmicNetwork

class RhythmicModel(GaussianTransitionModel):

    input_dims: int = Property(default=3, doc="input dimensions of tracked state")
    rhythmic_network: Type[RhythmicNetwork] = Property(doc="network for prediction")

    @property
    def ndim_state(self):
        return self.input_dims

    def function(self, state, noise=False, **kwargs) -> StateVector:
        if state.state_vector.shape[1] == 1:
            self.rhythmic_network.advance(state.state_vector[:,0], save_history=True)
        return state.state_vector

    def covar(self, time_interval, **kwargs):
        return CovarianceMatrix(5e10*np.identity(self.ndim_state))
