# Functions to add detectors not avaliable in generic driftbench
# Imports
import numpy as np
from scipy.stats import theilslopes
from scipy.ndimage import median_filter
from river.drift import ADWIN, binary, KSWIN, PageHinkley
from driftbench.drift_detection.detectors import Detector


class RiverDetector(Detector):
	def __init__(self, detector, **kwargs):
		self.detector_class = detector
		self.kwargs = kwargs

		self.hparams = list(kwargs.keys())
		for key, value in kwargs.items():
			setattr(self, key, value)

	def predict(self, X):	
		num_chans = X.shape[1]
		y = np.zeros_like(X)
		detectors = [self.detector_class(**self.kwargs) for _ in range(num_chans)]

		for i in range(num_chans):
			for t in range(X.shape[0]):
				detectors[i].update(X[t, i])
				if detectors[i].drift_detected:
					y[t:, i] = 1

		return np.sum(y, axis=1)/num_chans

	@property
	def name(self):
		return self.detector_class.__name__


class RiverDetectorSingleDetect(Detector):
	def __init__(self, detector, **kwargs):
		self.detector = detector(**kwargs)
		self.hparams = list(kwargs.keys())
		for key, value in kwargs.items():
			setattr(self, key, value)

	def predict(self, X):
		num_chans = X.shape[1]
		y = np.zeros_like(X)

		for i in range(num_chans):
			for t in range(X.shape[0]):
				self.detector.update(X[t, i])
				if self.detector.drift_detected:
					y[t, i] = 1

		return np.sum(y, axis=1)/num_chans

	@property
	def name(self):
		return self.detector.__class__.__name__


class TheilSlopesDetector(Detector):
	def __init__(self, min_slope, window_size):
		self.hparams = ["min_slope", "window_size"]
		self.min_slope = min_slope
		self.window_size = window_size

	def predict(self, X):
		num_chans = X.shape[1]
		data = median_filter(X.T, size=self.window_size * 4, axes=[-1])
	
		expanded_data = np.zeros(data.shape + (self.window_size,))
		expanded_data[:, :, 0] = data
		for i in range(1, self.window_size):
			expanded_data[:, :, i] = np.roll(expanded_data[:, :, i-1], -1, axis=-1)
		
		expanded_data = expanded_data[:, :-self.window_size + 1]
		stds = np.std(expanded_data, axis=-1)
		
		slopes = theilslopes(expanded_data, axis=-1)
		slopes, ub, lb = slopes.slope, slopes.low_slope, slopes.high_slope
	
		alarms = np.sum(slopes > self.min_slope, axis=0)
		# alarms = np.sum(np.logical_or(
		#	 np.logical_and(slopes > min_slope, lb > min_slope), np.logical_and(slopes < -min_slope, ub < -min_slope)
		# ), axis=0)
		# alarms = calc_river_alarms(KSWIN, slopes)
	
		return np.concatenate([np.zeros((X.shape[0] - len(alarms),)), alarms])
