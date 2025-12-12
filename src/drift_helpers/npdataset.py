# Function to create driftbench-usable data from np.array driftbench
# Imports
import numpy as np
from driftbench.benchmarks.data import Dataset


class NpDataset(Dataset):
	def __init__(self, name, data, y):
		self.spec = {}
		self.name = name
		self.n_variations = 1

		assert data.shape[0] > data.shape[1] # Check that there are more samples than channels
		self.data = data

		assert self.data.shape[0] == y.shape[0] # Check that the data and the labels are the same length
		assert len(y.shape) == 1 # Check y only has 1 channel
		# This is not checked but, if a drift starts at t=1 and ends at t=3, y should be 0111000 not 0100000 or 0101000
		self.y = y

	def __iter__(self):
		for i in range(self.data.shape[0]):
			yield i, self.data[i], self.y[i]
