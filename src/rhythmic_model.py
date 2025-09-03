from stonesoup.models.transition.nonlinear import GaussianTransitionModel
from stonesoup.types.state import StateVector, StateVectors
from stonesoup.types.array import CovarianceMatrix
from stonesoup.base import Property
import numpy as np

class RhythmicModel(GaussianTransitionModel):

    input_dims: int = Property(default=3, doc="input dimensions of tracked state")

    @property
    def ndim_state(self):
        return self.input_dims

    def function(self, state, noise=False, **kwargs) -> StateVector:
        return 2*state.state_vector

    def covar(self, time_interval, **kwargs):
        return CovarianceMatrix(np.identity(self.ndim_state))
