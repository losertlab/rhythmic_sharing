# Functions to calculate metrics not avaliable in generic driftbench
# Imports
from driftbench.drift_detection.metrics import Metric
import numpy as np


class BaseConfusionMatrixMetric(Metric):
	def __init__(self, max_delay=None, return_all=False):
		self.max_delay = max_delay
		self.return_all = return_all

	def __call__(self, prediction, targets):
		
		# Set max_delay from shortest real drift
		drift_changes = np.where(np.abs(targets[1:] - targets[:-1]) > 0)[0]
		if not self.max_delay and len(drift_changes) >= 2:
			shortest_drift = np.min(np.diff(drift_changes)[::2])
			self.max_delay = shortest_drift

		# Get discrete starts/ends and setup loop
		discrete_drift_starts = drift_changes[0::2]
		discrete_drift_ends = drift_changes[1::2]
		drift_starts_ends = zip(discrete_drift_starts, discrete_drift_ends)

		thresholds = np.unique(prediction)
		metrics = []

		# Loop through calculating metric
		for t in thresholds:
			discrete_predictions = np.where(np.diff(prediction > t) > 0)[0] - 1
			metrics.append(self.run_metric(drift_starts_ends, discrete_predictions))

		# Choose which metric to return
		if self.return_all:
			return metrics
		return self.choose_which(metrics)

	def run_metric(self, drift_starts_ends, discrete_predictions):
		raise NotImplementedError

	def choose_which(self, metric_results):
		raise NotImplementedError


class FP(BaseConfusionMatrixMetric):
	def run_metric(self, drift_starts_ends, discrete_predictions):
		FPs = 0
		for pred in discrete_predictions:
			fp = True
			for start, end in drift_starts_ends:
				if start <= pred <= end and pred - start < self.max_delay:
					fp = False
					break
					
			if fp:
				FPs += 1

		return FPs

	def choose_which(self, metric_results):
		return min(metric_results)


class TP(BaseConfusionMatrixMetric):
	def run_metric(self, drift_starts_ends, discrete_predictions):
		TPs = 0
		for pred in discrete_predictions:
			for start, end in drift_starts_ends:
				if start <= pred <= end and pred - start < self.max_delay:
					TPs += 1
					break

		return TPs

	def choose_which(self, metric_results):
		return max(metric_results)


class FN(BaseConfusionMatrixMetric):
	def run_metric(self, drift_starts_ends, discrete_predictions):
		FNs = 0
		for start, end in drift_starts_ends:
			no_detect = True
			for pred in discrete_predictions:
				if start <= pred <= end and pred - start < self.max_delay:
					no_detect = False
					break
					
			if no_detect:
				FNs += 1

		return FNs

	def choose_which(self, metric_results):
		return min(metric_results)


class Precision(BaseConfusionMatrixMetric):
	def __call__(self, prediction, targets):
		TPs = np.asarray(TP(max_delay=self.max_delay, return_all=True)(prediction, targets))
		FPs = np.asarray(FP(max_delay=self.max_delay, return_all=True)(prediction, targets))

		if self.return_all:
			return TPs/(TPs + FPs)
		return max(TPs/(TPs + FPs))


class Recall(BaseConfusionMatrixMetric):
	def __call__(self, prediction, targets):
		TPs = np.asarray(TP(max_delay=self.max_delay, return_all=True)(prediction, targets))
		FNs = np.asarray(FN(max_delay=self.max_delay, return_all=True)(prediction, targets))

		if self.return_all:
			return TPs/(TPs + FNs)
		return max(TPs/(TPs + FNs))


class F1_Score(BaseConfusionMatrixMetric):
	def __call__(self, prediction, targets):
		Ps = np.asarray(Precision(max_delay=self.max_delay, return_all=True)(prediction, targets))
		Rs = np.asarray(Recall(max_delay=self.max_delay, return_all=True)(prediction, targets))

		if self.return_all:
			return (2 * Ps * Rs)/(Ps + Rs)
		return max((2 * Ps * Rs)/(Ps + Rs))


class Delay(BaseConfusionMatrixMetric):
	def run_metric(self, drift_starts_ends, discrete_predictions):
		delays = []
		for pred in discrete_predictions:
			for start, end in drift_starts_ends:
				if start <= pred <= end and pred - start < self.max_delay:
					delays.append(pred - start)
					break

		return np.mean(delays)

	def choose_which(self, metric_results):
		return min(metric_results)
