from .generic import GenericProbe, IChannel
import numpy as np

from pymeasure.instruments.instrument import Instrument
from pymeasure.instruments.oscilloscope import Oscilloscope
from pymeasure.adapters import VISAAdapter


class Probe(GenericProbe):
	def __init__(self, *args, **kwargs):
		if not args:
			args = [0]
		self.args = args
		self.kwargs = kwargs
	
	def open(self):
		self.inst = Oscilloscope(*self.args, **self.kwargs)
		self.timeChannel = self.inst["T"]
		#print(chans)
		#todo:
	
	@property
	def chans(self):
		return self.inst.channels

	@property
	def RATE(self):
		res = None
		for c in self.inst.channels:
			if c is not self.timeChannel:
				print("c", c, c.__class__.mro())
				if res is None:
					res = c.get_samples_per_channel_sample(self.timeChannel)
				else:
					if res != c:
						c.set_samples_per_channel_sample(self.timeChannel, res)
		return res

	@RATE.setter
	def RATE(self, v):
		for c in self.inst.channels:
			c.set_samples_per_channel_sample(self.timeChannel, v)

	@property
	def CHUNK(self):
		res = None
		for c in self.inst.channels:
			if res is None:
				res = c.memory_size
			else:
				if res != c:
					c.memory_size = res
		return res

	@CHUNK.setter
	def CHUNK(self, v):
		for c in self.inst.channels:
			c.memory_size = v

	def read(self, channels, npoints, verbose=False):
		acq = WaveformSynchronisedAcquisitor(self.inst, channels)
		return acq()

	def close(self):
		pass
