# Functions to calculate metrics not avaliable in generic driftbench
# Imports
from driftbench.drift_detection.metrics import Metric
import numpy as np


class BaseConfusionMatrixMetric(Metric):
	def __init__(self, return_all=False, f1_metric_obj=None, set_threshold=None):
		self.return_all = return_all
		self.f1_metric_obj = f1_metric_obj

	def __call__(self, prediction, targets, set_threshold=None):
		
		self.thresholds = np.unique(prediction[prediction > 0])
		if set_threshold:
			self.thresholds = [set_threshold]
		elif len(self.thresholds) > 1000:
			self.thresholds = np.unique(self.reduce_thresholds(prediction))
		metrics = []

		# Loop through calculating metric
		for t in self.thresholds:
			metrics.append(self.run_metric(prediction >= t, targets))

		# Choose which metric to return
		if self.return_all:
			return metrics
		elif self.f1_metric_obj:
			return metrics[self.f1_metric_obj.chosen]
		return self.choose_which(metrics)

	def reduce_thresholds(self, prediction):
		prediction = (prediction - np.min(prediction))/(np.max(prediction) - np.min(prediction))
		return np.round(prediction, decimals=3)
	
	def run_metric(self, prediction, targets):
		raise NotImplementedError

	def choose_which(self, metric_results):
		raise NotImplementedError


class FP(BaseConfusionMatrixMetric):
	def run_metric(self, prediction, targets):
		return np.sum(np.logical_and(targets == 0, prediction == 1).astype(float))

	def choose_which(self, metric_results):
		return min(metric_results)


class TP(BaseConfusionMatrixMetric):
	def run_metric(self, prediction, targets):
		return np.sum(np.logical_and(targets == 1, prediction == 1).astype(float))

	def choose_which(self, metric_results):
		return max(metric_results)


class FN(BaseConfusionMatrixMetric):
	def run_metric(self, prediction, targets):
		return np.sum(np.logical_and(targets == 1, prediction == 0).astype(float))

	def choose_which(self, metric_results):
		return min(metric_results)


class Precision(BaseConfusionMatrixMetric):
	def __call__(self, prediction, targets, set_threshold=None):
		TPs = np.asarray(TP(return_all=True)(prediction, targets, set_threshold=set_threshold))
		FPs = np.asarray(FP(return_all=True)(prediction, targets, set_threshold=set_threshold))

		if self.return_all:
			return TPs/(TPs + FPs)
		elif self.f1_metric_obj:
			return (TPs/(TPs + FPs))[self.f1_metric_obj.chosen]
		return max(TPs/(TPs + FPs))


class Recall(BaseConfusionMatrixMetric):
	def __call__(self, prediction, targets, set_threshold=None):
		TPs = np.asarray(TP(return_all=True)(prediction, targets, set_threshold=set_threshold))
		FNs = np.asarray(FN(return_all=True)(prediction, targets, set_threshold=set_threshold))

		if self.return_all:
			return TPs/(TPs + FNs)
		elif self.f1_metric_obj:
			return (TPs/(TPs + FNs))[self.f1_metric_obj.chosen]
		return max(TPs/(TPs + FNs))


class F1_Score(BaseConfusionMatrixMetric):
	def __call__(self, prediction, targets, set_threshold=None):
		Ps = np.asarray(Precision(return_all=True)(prediction, targets, set_threshold=set_threshold))
		Rs = np.asarray(Recall(return_all=True)(prediction, targets, set_threshold=set_threshold))
		F1s = (2 * Ps * Rs)/(Ps + Rs)

		if self.return_all:
			return F1s

		self.chosen = np.nanargmax(F1s)
		tp = TP()
		tp(prediction, targets, set_threshold=set_threshold)
		self.thresh = tp.thresholds[self.chosen]

		return F1s[self.chosen]

class Accuracy(BaseConfusionMatrixMetric):
	def __call__(self, prediction, targets, set_threshold=None):
		FNs = np.asarray(FN(return_all=True)(prediction, targets, set_threshold=set_threshold))
		FPs = np.asarray(FN(return_all=True)(prediction, targets, set_threshold=set_threshold))
		Ts = len(prediction) - (FNs + FPs)

		if self.return_all:
			return Ts/len(prediction)
		elif self.f1_metric_obj:
			return (Ts/len(prediction))[self.f1_metric_obj.chosen]
		return max(Ts/len(prediction))	

class Delay(BaseConfusionMatrixMetric):
	def run_metric(self, prediction, targets):
		drift_changes = np.where(np.abs(targets[1:] - targets[:-1]) > 0)[0]

		discrete_drift_starts = drift_changes[0::2].tolist()
		discrete_drift_ends = drift_changes[1::2].tolist()
		if len(discrete_drift_starts) == len(discrete_drift_ends) + 1:
			discrete_drift_ends += [len(targets)]
		drift_starts_ends = zip(discrete_drift_starts, discrete_drift_ends)

		first_detections = []
		for start, end in drift_starts_ends:
			# Get the first detection
			first_detection = np.where(prediction[start:end])[0]
			if len(first_detection) > 0:
				first_detections.append(np.min(first_detection))

		if len(first_detections) > 0:
			return np.mean(first_detections)
		return 0

	def choose_which(self, metric_results):
		return np.nanmin(metric_results)

class EarlyDetect(BaseConfusionMatrixMetric):
	def run_metric(self, prediction, targets):
		drift_changes = np.where(np.abs(targets[1:] - targets[:-1]) > 0)[0]

		discrete_drift_starts = drift_changes[0::2].tolist()
		discrete_drift_ends = drift_changes[1::2].tolist()
		if len(discrete_drift_starts) == len(discrete_drift_ends) + 1:
			discrete_drift_ends += [len(targets)]
		drift_starts_ends = [(None, 0)] + list(zip(discrete_drift_starts, discrete_drift_ends))

		early_detections = []
		for i in range(1, len(drift_starts_ends)):
			# Pretend a drift starts at t=10 but the detector predicts it by setting p_{t_{7:10}} to 1
			# Then, we should add 3 (10-7) to early detections
			# The idea is to get the time of the earliest measurement that occurs after a previous drift and a drift is predicted between that time and the start of the drift

			prev_drift_end = drift_starts_ends[i-1][1]
			cur_drift_start = drift_starts_ends[i][0]
			detections = np.where(prediction[prev_drift_end:cur_drift_start+1])[0]
			# print(prediction[prev_drift_end:cur_drift_start+1], detections)
			if len(detections) > 0:
				earliest = None
				for detection_time in detections[::-1]:
					# print("Trying", detection_time, prediction[prev_drift_end+detection_time:cur_drift_start+1])
					if np.all(prediction[prev_drift_end+detection_time:cur_drift_start+1]):
						earliest = cur_drift_start - (prev_drift_end + detection_time)
					else:
						break
				if earliest:
					early_detections.append(earliest)
				# 	print(earliest)
				# else:
				# 	print("NO")
		
		if len(early_detections) > 0:
			return np.mean(early_detections)
		return 0

	def choose_which(self, metric_results):
		return np.nanmax(metric_results)