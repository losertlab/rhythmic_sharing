from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.types.state import StateVector, StateVectors
from stonesoup.types.array import CovarianceMatrix
from stonesoup.base import Property
import numpy as np
from typing import Type

from rhythmic_sharing import RhythmicNetwork

class RhythmicModel(GaussianTransitionModel):

    dims: int = Property(default=3, doc="input dimensions of tracked state")
    rhythmic_network: Type[RhythmicNetwork] = Property(doc="network for prediction")

    @property
    def ndim_state(self):
        return self.dims

    def function(self, state, noise=False, **kwargs) -> StateVector:
        vecs = np.copy(state.state_vector.T)
        new_state = np.zeros(vecs.shape)
        for idx, state_vector in enumerate(vecs):
            self.rhythmic_network.advance_static(state_vector)
            new_state[idx, :] = self.rhythmic_network.get_output(static=True)
        return StateVectors(new_state.T)

    def covar(self, time_interval, **kwargs):
        pass
        sigma_0 = 0.1
        return CovarianceMatrix(sigma_0*np.identity(self.ndim_state))
