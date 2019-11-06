#!/usr/bin/env python

"""
Oscilloscope + spectrum analyser in Python for the NIOS server.

Modified version from the original code by R. Fearick.

Giuseppe Venturini, July 2012-2013

Original copyright notice follows. The same license applies.

------------------------------------------------------------
Copyright (C) 2008, Roger Fearick, University of Cape Town

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
------------------------------------------------------------

Version 0.1

This code provides a two-channel oscilloscope and spectrum analyzer.

Dependencies:
Python 2.6+
numpy         -- numerics, fft
PyQt4, PyQwt5 -- gui, graphics

Optional packages:
pyspectrum    -- expert mode spectrum calculation

Typically, a modification of the Python path and ld library is necessary,
like this:
	  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
	  export PYTHONPATH=$PYTHONPATH:.

The code can be adjusted for different sampling rates and chunks lengths.

The interface, based on qwt, uses a familar 'knob based' layout so that it
approximates an analogue scope.

Traces can be averaged to reduce influence of noise.
A cross hair status display permits the reading of values off the screen.
Printing and exporting CSV and PDF files are provided.


FFT options

- by default we use the periogram algorithm from pyspectrum [1] - not
  in Debian stable but available through pypi and easy_install.
  [1] https://www.assembla.com/spaces/PySpectrum/wiki

- If 'pyspectrum' is not available, we fallback to using the FFT
 method from numpy to compute the PSD.

- Using numpy to calculate the FFT can be forced setting:
   USE_NUMPY_FFT = True
  in the following code.

- additionally, it is possible to use matplotlib.psd().
  -> you need to modify the sources to do so.


INSTALLING pyspectrum

The package pyspectrum can be installed with either:
 'pip install spectrum'

"""
import sys
import struct
import subprocess
import time
import os.path
import configparser
import importlib
from collections import defaultdict, OrderedDict
from functools import partial

from PyQt5 import Qt
from PyQt5 import QtWidgets
import qwt as Qwt

import numpy as np
import numpy.fft as FFT

from scipy.interpolate import InterpolatedUnivariateSpline

# part of this package -- csv interface and toolbar icons
from . import csvlib, icons, utils
import dualscope123.probes

# from dualscope123.probes import eth_nios

# scope configuration
CHANNELS = 2
DEFAULT_TIMEBASE = 0.01
BOTH12 = 0
CH1 = 1
CH2 = 2
scopeheight = 500  # px
scopewidth = 800  # px
SELECTEDCH = BOTH12
TIMEPENWIDTH = 1
FFTPENWIDTH = 2

# status messages
freezeInfo = "Freeze: Press mouse button and drag"
cursorInfo = "Cursor Pos: Press mouse button in plot region"

# FFT CONFIG
USE_NUMPY_FFT = False
try:
	import spectrum

	print("(II) spectrum MODULE FOUND")
	SPECTRUM_MODULE = True
except ImportError:
	print("(WW) PSD: spectrum MODULE NOT FOUND")
	SPECTRUM_MODULE = False
if USE_NUMPY_FFT:
	print("(WW) SPECTRUM MODULE DISABLED in source")
	SPECTRUM_MODULE = False
if not SPECTRUM_MODULE:
	print("(WW) PSD: using FFTs through NUMPY.fftpack")

# PyQt5.QtWidgets.QSpinBox
# utility classes


class LogKnob(QtWidgets.QDoubleSpinBox):
	"""
	Provide knob with log scale
	"""

	def __init__(self, *args):
		print(args)
		super().__init__(*args)
		# self.setScaleEngine(Qwt.QwtLog10ScaleEngine())

	def setRange(self, minR, maxR, step=0.01):
		# self.setScale(minR, maxR)
		#super().setRange(minR, maxR)
		super().setRange(np.log10(minR), np.log10(maxR))
		super().setSingleStep(step)
		#super().setRange(np.log10(minR), np.log10(maxR), step)

	def setValue(self, val):
		super().setValue(np.log10(val))


class LblKnob:
	"""
	Provide knob with a label
	"""

	def __init__(self, wgt, x, y, name, logscale=0):
		if logscale:
			self.knob = LogKnob(wgt)
		else:
			self.knob = QtWidgets.QDoubleSpinBox(wgt)
		self.knob.setSingleStep(0.001)
		color = Qt.QColor(200, 200, 210)
		self.knob.palette().setColor(Qt.QPalette.Active, Qt.QPalette.Button, color)
		self.lbl = Qt.QLabel(name, wgt)
		self.knob.setGeometry(x, y, 140, 100)
		# oooh, eliminate this ...
		if name[0] == "o":
			# self.knob.setKnobWidth(40)
			self.knob.setMinimumWidth(40)
		self.lbl.setGeometry(x, y + 90, 140, 15)
		self.lbl.setAlignment(Qt.Qt.AlignCenter)

	def setRange(self, max, min, step=None):
		self.knob.setRange(max, min)
		if step:
			self.knob.setSingleStep(step)

	def setValue(self, v):
		self.knob.setValue(v)

	def setScaleMaxMajor(self, *args):
		# self.knob.setScaleMaxMajor(*args)
		pass

class ComponentsStorage:
	__slots__ = ("comps",)
	def __init__(self, compsCount=3):
		self.comps = [None for i in range(compsCount)]
	
	def __getitem__(self, k):
		return self.comps[k]

	def __setitem__(self, k, v):
		self.comps[k] = v
	
	def __iter__(self):
		yield from range(len(self.comps))
	
	def __contains__(self, k):
		return 0<=k<=len(self.comps)
	
	def __len__(self):
		return len(self.comps)
	
	@property
	def x(self):
		return self.comps[0]
	
	@x.setter
	def x(self):
		return self.comps[0]

	@property
	def y(self):
		return self.comps[1]
	
	@y.setter
	def y(self):
		return self.comps[1]

	@property
	def z(self):
		return self.comps[2]
	
	@z.setter
	def z(self):
		return self.comps[2]

class CurveSource(ComponentsStorage):
	__slots__ = ()
	"""All the coordinates must be from the single device!"""

class DataStorage(ComponentsStorage):
	__slots__ = ()
	"""Data for the channels"""
	def __len__(self):
		return len(tuple(iter(self)))
	
	def __iter__(self):
		for i in super().__iter__():
			if self[i] is not None:
				yield i
	
	def __contains__(self, k):
		return super().__contains__(k) and self[k] is not None


class Curve():
	__slots__ = ("curve", "symbol", "pen", "maxamp", "offset")
	def __init__(self, parentPlot, name):
		self.offset = 0.
		self.maxamp = 5.0
		self.curve = Qwt.QwtPlotCurve(name)
		self.symbol = Qwt.QwtSymbol(Qwt.QwtSymbol.Ellipse, Qt.QBrush(2), Qt.QPen(Qt.Qt.darkMagenta), Qt.QSize(3, 3))
		self.pen = Qt.QPen(Qt.Qt.magenta, TIMEPENWIDTH)
		self.curve.setSymbol(self.symbol)
		self.curve.setPen(self.pen)
		self.curve.setYAxis(Qwt.QwtPlot.yRight)
		self.curve.attach(parentPlot)
	
	def plot(self, res):
		d = len(res)
		if d == 2:
			#todo: self.a2 + self.offset * self.maxamp
			#print(res[0], res[1])
			self.curve.setData(res[0], res[1])
		elif d == 0:
			return
		else:
			raise NotImplementedError(str(d) + "-dimensional plots are not implemented yet")

class Scope(Qwt.QwtPlot):
	"""
	Oscilloscope display widget
	"""

	def __init__(self, *args):
		super().__init__(*args)
		
		self.curvesSources = [CurveSource() for i in range(2)]
		self.plots = {id(cs):Curve(self, "Trace "+str(i)) for i, cs in enumerate(self.curvesSources)}

		self.setTitle("Scope")
		self.setCanvasBackground(Qt.Qt.white)

		# grid
		self.grid = Qwt.QwtPlotGrid()
		self.grid.enableXMin(True)
		self.grid.setMajPen(Qt.QPen(Qt.Qt.gray, 0, Qt.Qt.SolidLine))
		self.grid.attach(self)

		# axes
		self.enableAxis(Qwt.QwtPlot.yRight)
		self.setAxisTitle(Qwt.QwtPlot.xBottom, "Time [s]")
		self.setAxisTitle(Qwt.QwtPlot.yLeft, "Amplitude []")
		self.setAxisMaxMajor(Qwt.QwtPlot.xBottom, 10)
		self.setAxisMaxMinor(Qwt.QwtPlot.xBottom, 0)

		self.setAxisScaleEngine(Qwt.QwtPlot.yRight, Qwt.QwtLinearScaleEngine())
		self.setAxisMaxMajor(Qwt.QwtPlot.yLeft, 10)
		self.setAxisMaxMinor(Qwt.QwtPlot.yLeft, 0)
		self.setAxisMaxMajor(Qwt.QwtPlot.yRight, 10)
		self.setAxisMaxMinor(Qwt.QwtPlot.yRight, 0)

		# default settings
		self.triggerval = 0.10
		self.triggerCH = None
		self.triggerslope = 0
		self.freeze = 0
		self.average = 0
		self.autocorrelation = 0
		self.avcount = 0
		self.datastream = None
		self.maxtime = 0.1

		# set data
		# NumPy: f, g, a and p are arrays!
		print("samplerate", samplerate)
		self.dt = 1.0 / samplerate
		interpolationSize = 1000
		
		#InterpolatedUnivariateSpline()

		# start self.timerEvent() callbacks running
		self.timer_id = self.startTimer(self.maxtime * 100 + 50)
		# plot
		self.replot()
		self.repaint()

	# convenience methods for knob callbacks
	def setMaxAmp(self, val):
		self.plots[id(self.curvesSources[0])].maxamp = val

	def setMaxAmp2(self, val):
		self.plots[id(self.curvesSources[1])].maxamp = val

	def setMaxTime(self, val):
		self.maxtime = val

	def setOffset1(self, val):
		self.plots[id(self.curvesSources[0])].offset = val

	def setOffset2(self, val):
		self.plots[id(self.curvesSources[1])].offset = val

	def setTriggerLevel(self, val):
		self.triggerval = val

	def setTriggerCH(self, val):
		self.triggerCH = val

	def setTriggerSlope(self, val):
		self.triggerslope = val

	# plot scope traces
	def setDisplay(self):
		#print("self.plotData", self.plotData)
		for curveSource in self.curvesSources:
			csId = id(curveSource)
			res = self.plotData[csId]
			self.plots[csId].plot(res)
		
		self.replot()
		self.repaint()

	def getValue(self, index):
		return self.f[index], self.a[index]

	def setAverage(self, state):
		self.average = state
		self.avcount = 0

	def setAutoc(self, state):
		self.autocorrelation = state
		self.avcount = 0

	def setFreeze(self, freeze):
		self.freeze = freeze

	def setDatastream(self, datastream):
		self.datastream = datastream

	def updateTimer(self):
		self.killTimer(self.timer_id)
		self.timer_id = self.startTimer(self.maxtime * 100 + 50)

	# timer callback that does the work
	def timerEvent(self, e):  # Scope
		global fftbuffersize
		if self.datastream is None:
			return
		if self.freeze == 1:
			return
		points = int(np.ceil(self.maxtime * samplerate))
		if self.triggerCH or self.autocorrelation:
			# we read twice as much data to be sure to be able to display data for all time points.
			# independently of trigger point location.
			read_points = 2 * points
		else:
			read_points = points
		fftbuffersize = read_points
		print("fftbuffersize set read_points", fftbuffersize)
		
		csId = {}
		csRes = {}
		sId = {}
		spreading = defaultdict(list)
		curves2get = []
		print("self.curvesSources", self.curvesSources)
		for cs in self.curvesSources:
			csId[id(cs)] = cs
			csRes[id(cs)] = DataStorage()
			for j in cs:
				s = cs[j]
				print(j, s)
				if s:
					curves2get.append(id(s))
					sId[id(s)] = s
					spreading[id(s)].append((id(cs), j))
		
		curves2getIds = tuple(set(curves2get))
		curves2get = tuple(sId[si] for si in curves2getIds)
		#print("curves2get", curves2get)
		streamsByChannels = self.datastream.read(curves2get, read_points, verbose)
		#print("streamsByChannels", streamsByChannels)
		for si, data in zip(curves2getIds, streamsByChannels):
			#print("si, data", si, data)
			for csi, j in spreading[si]:
				csRes[csi][j] = data
		
		self.plotData = csRes
		self.setDisplay()
		return

		if self.triggerCH == 1 and (SELECTEDCH == BOTH12 or SELECTEDCH == CH1):
			print("Waiting for CH1 trigger...")
			if self.triggerslope == 0:
				zero_crossings = np.where(np.diff(np.sign(data_CH1[points / 2 : -points / 2] - self.triggerval * self.plots[id(self.curvesSources[0])].maxamp)) != 0)[0]
			if self.triggerslope == 1:
				zero_crossings = np.where(np.diff(np.sign(data_CH1[points / 2 : -points / 2] - self.triggerval * self.plots[id(self.curvesSources[0])].maxamp)) > 0)[0]
			if self.triggerslope == 2:
				zero_crossings = np.where(np.diff(np.sign(data_CH1[points / 2 : -points / 2] - self.triggerval * self.plots[id(self.curvesSources[0])].maxamp)) < 0)[0]
			if not len(zero_crossings):
				return
			print("Triggering on sample", zero_crossings[0])
			imin = zero_crossings[0]
			imax = zero_crossings[0] + points
			data_CH1 = data_CH1[imin:imax]
		elif self.triggerCH == 2 and (SELECTEDCH == BOTH12 or SELECTEDCH == CH2):
			print("Waiting for CH2 trigger...")
			if self.triggerslope == 0:
				zero_crossings = np.where(np.diff(np.sign(data_CH2[points / 2 : -points / 2] - self.triggerval * self.plots[id(self.curvesSources[1])].maxamp)) != 0)[0]
			if self.triggerslope == 1:
				zero_crossings = np.where(np.diff(np.sign(data_CH2[points / 2 : -points / 2] - self.triggerval * self.plots[id(self.curvesSources[1])].maxamp)) > 0)[0]
			if self.triggerslope == 2:
				zero_crossings = np.where(np.diff(np.sign(data_CH2[points / 2 : -points / 2] - self.triggerval * self.plots[id(self.curvesSources[1])].maxamp)) < 0)[0]
			if not len(zero_crossings):
				return
			print("Triggering on sample", zero_crossings[0])
			imin = zero_crossings[0]
			imax = zero_crossings[0] + points
			data_CH2 = data_CH2[imin:imax]

		if self.autocorrelation:
			if SELECTEDCH == BOTH12 or SELECTEDCH == CH1:
				data_CH1 = utils.autocorrelation(data_CH1[: 2 * points])[:points]
			else:
				data_CH1 = np.zeros((points,))
			if SELECTEDCH == BOTH12 or SELECTEDCH == CH2:
				data_CH2 = utils.autocorrelation(data_CH2[: 2 * points])[:points]
			else:
				data_CH2 = np.zeros((points,))

		if self.average == 0:
			self.a1 = data_CH1
			self.a2 = data_CH2
		else:
			self.avcount += 1
			if self.avcount == 1:
				self.sumCH1 = np.array(data_CH1, dtype=np.float_)
				self.sumCH2 = np.array(data_CH2, dtype=np.float_)
			else:
				if SELECTEDCH == BOTH12:
					assert len(data_CH1) == len(data_CH2)
					lp = len(data_CH1)
					if len(self.sumCH1) == lp and len(self.sumCH2) == lp:
						self.sumCH1 = self.sumCH1[:lp] + np.array(data_CH1[:lp], dtype=np.float_)
						self.sumCH2 = self.sumCH2[:lp] + np.array(data_CH2[:lp], dtype=np.float_)
					else:
						self.sumCH1 = np.array(data_CH1, dtype=np.float_)
						self.sumCH2 = np.array(data_CH2, dtype=np.float_)
						self.avcount = 1
				elif SELECTEDCH == CH1:
					lp = len(data_CH1)
					if len(self.sumCH1) == lp:
						self.sumCH1 = self.sumCH1[:lp] + np.array(data_CH1[:lp], dtype=np.float_)
					else:
						self.sumCH1 = np.array(data_CH1, dtype=np.float_)
						self.avcount = 1
				elif SELECTEDCH == CH2:
					lp = len(data_CH2)
					if len(self.sumCH2) == lp:
						self.sumCH2 = self.sumCH2[:lp] + np.array(data_CH2[:lp], dtype=np.float_)
					else:
						self.sumCH2 = np.array(data_CH2, dtype=np.float_)
						self.avcount = 1

			self.a1 = self.sumCH1 / self.avcount
			self.a2 = self.sumCH2 / self.avcount
		self.setDisplay()


inittime = 0.01
initamp = 100


class ScopeFrame(Qt.QFrame):
	"""
	Oscilloscope widget --- contains controls + display
	"""

	def __init__(self, *args):
		super().__init__(*args)
		# the following: setPal..  doesn't seem to work on Win
		try:
			self.setPaletteBackgroundColor(QColor(240, 240, 245))
		except BaseException:
			pass
		hknobpos = scopewidth + 20
		vknobpos = scopeheight + 30
		self.setFixedSize(scopewidth + 150, scopeheight + 150)
		self.freezeState = 0
		self.triggerComboBox = Qt.QComboBox(self)
		self.triggerComboBox.setGeometry(hknobpos + 10, 50, 100, 40)  # "Channel: ")
		self.triggerComboBox.addItem("Trigger off")
		self.triggerComboBox.addItem("CH1")
		self.triggerComboBox.addItem("CH2")
		self.triggerComboBox.setCurrentIndex(0)
		self.triggerSlopeComboBox = Qt.QComboBox(self)
		self.triggerSlopeComboBox.setGeometry(hknobpos + 10, 100, 100, 40)  # "Channel: ")
		self.triggerSlopeComboBox.addItem("Any Slope")
		self.triggerSlopeComboBox.addItem("Positive")
		self.triggerSlopeComboBox.addItem("Negative")
		self.triggerSlopeComboBox.setCurrentIndex(0)
		self.knbLevel = LblKnob(self, hknobpos, 160, "Trigger level (%FS)")
		self.knbTime = LblKnob(self, hknobpos, 300, "Time", 1)
		self.knbSignal = LblKnob(self, 150, vknobpos, "Signal1", 1)
		self.knbSignal2 = LblKnob(self, 450, vknobpos, "Signal2", 1)
		self.knbOffset1 = LblKnob(self, 10, vknobpos, "offset1")
		self.knbOffset2 = LblKnob(self, 310, vknobpos, "offset2")

		self.knbTime.setRange(0.0001, 1.0)
		self.knbTime.setValue(DEFAULT_TIMEBASE)

		self.knbSignal.setRange(1, 1e6, 1)
		self.knbSignal.setValue(100.0)

		self.knbSignal2.setRange(1, 1e6, 1)
		self.knbSignal2.setValue(100.0)

		self.knbOffset2.setRange(-1.0, 1.0, 0.1)
		self.knbOffset2.setValue(0.0)

		self.knbOffset1.setRange(-1.0, 1.0, 0.1)
		self.knbOffset1.setValue(0.0)

		self.knbLevel.setRange(-1.0, 1.0, 0.1)
		self.knbLevel.setValue(0.1)
		# self.knbLevel.setScaleMaxMajor(10)

		self.plot = Scope(self)
		self.plot.setGeometry(10, 10, scopewidth, scopeheight)
		# self.picker = Qt.QColorDialog(
		#    Qwt.QwtPlot.xBottom,
		#    Qwt.QwtPlot.yLeft,
		#    Qwt.QwtPicker.PointSelection | Qwt.QwtPicker.DragSelection,
		#    Qwt.QwtPlotPicker.CrossRubberBand,
		#    Qwt.QwtPicker.ActiveOnly, #AlwaysOn,
		#    self.plot.canvas()
		# )
		# self.picker.setRubberBandPen(Qt.QPen(Qt.Qt.green))
		# self.picker.setTrackerPen(Qt.QPen(Qt.Qt.cyan))

		self.knbTime.knob.valueChanged.connect(self.setTimebase)
		self.knbTime.setValue(0.01)
		self.knbSignal.knob.valueChanged.connect(self.setAmplitude)
		self.knbSignal2.knob.valueChanged.connect(self.setAmplitude2)
		#self.knbSignal.setValue(0.1)
		self.knbLevel.knob.valueChanged.connect(self.setTriggerlevel)
		self.knbOffset1.knob.valueChanged.connect(self.plot.setOffset1)
		self.knbOffset2.knob.valueChanged.connect(self.plot.setOffset2)
		self.triggerComboBox.currentIndexChanged.connect(self.setTriggerCH)
		self.triggerSlopeComboBox.currentIndexChanged.connect(self.plot.setTriggerSlope)
		self.knbLevel.setValue(0.1)
		self.plot.setAxisScale(Qwt.QwtPlot.xBottom, 0.0, 10.0 * inittime)
		self.plot.setAxisScale(Qwt.QwtPlot.yLeft, -initamp, initamp)
		self.plot.setAxisScale(Qwt.QwtPlot.yRight, -initamp, initamp)
		self.plot.show()

	def _calcKnobVal(self, val):
		ival = np.floor(val)
		"""frac = val - ival
		if frac >= 0.9:
			frac = 1.0
		elif frac >= 0.66:
			frac = np.log10(5.0)
		elif frac >= np.log10(2.0):
			frac = np.log10(2.0)
		else:
			frac = 0.0
		dt = 10 ** frac * 10 ** ival"""
		dt = 10 ** val
		return dt

	def setTimebase(self, val):
		dt = self._calcKnobVal(val)
		print("val", val)
		print("dt", dt)
		self.plot.setAxisScale(Qwt.QwtPlot.xBottom, 0.0, 10.0 * dt)
		self.plot.setMaxTime(dt * 10.0)
		print("ScopeFrame.setTimebase replotting")
		self.plot.replot()

	def setAmplitude(self, val):
		dt = self._calcKnobVal(val)
		self.plot.setAxisScale(Qwt.QwtPlot.yLeft, -dt, dt)
		self.plot.setMaxAmp(dt)
		print("ScopeFrame.setAmplitude replotting")
		self.plot.replot()

	def setAmplitude2(self, val):
		dt = self._calcKnobVal(val)
		self.plot.setAxisScale(Qwt.QwtPlot.yRight, -dt, dt)
		self.plot.setMaxAmp2(dt)
		print("ScopeFrame.setAmplitude2 replotting")
		self.plot.replot()

	def setTriggerlevel(self, val):
		self.plot.setTriggerLevel(val)
		self.plot.setDisplay()

	def setTriggerCH(self, val):
		if val == 0:
			val = None
		self.plot.setTriggerCH(val)
		self.plot.setDisplay()

	def setDatastream(self, v):
		self.plot.setDatastream(v)
# --------------------------------------------------------------------


class FScope(Qwt.QwtPlot):
	"""
	Power spectrum display widget
	"""

	def __init__(self, *args):
		super().__init__(*args)

		self.setTitle("Power spectrum")
		self.setCanvasBackground(Qt.Qt.white)

		# grid
		self.grid = Qwt.QwtPlotGrid()
		self.grid.enableXMin(True)
		self.grid.setMajPen(Qt.QPen(Qt.Qt.gray, 0, Qt.Qt.SolidLine))
		self.grid.attach(self)

		# axes
		self.setAxisTitle(Qwt.QwtPlot.xBottom, "Frequency [Hz]")
		self.setAxisTitle(Qwt.QwtPlot.yLeft, "Power Spectrum [dBc/Hz]")
		self.setAxisMaxMajor(Qwt.QwtPlot.xBottom, 10)
		self.setAxisMaxMinor(Qwt.QwtPlot.xBottom, 0)
		self.setAxisMaxMajor(Qwt.QwtPlot.yLeft, 10)
		self.setAxisMaxMinor(Qwt.QwtPlot.yLeft, 0)

		# curves
		self.curve1 = Qwt.QwtPlotCurve("PSTrace2")
		self.curve1.setPen(Qt.QPen(Qt.Qt.magenta, FFTPENWIDTH))
		self.curve1.setYAxis(Qwt.QwtPlot.yLeft)
		self.curve1.attach(self)

		self.curve0 = Qwt.QwtPlotCurve("PSTrace1")
		self.curve0.setPen(Qt.QPen(Qt.Qt.blue, FFTPENWIDTH))
		self.curve0.setYAxis(Qwt.QwtPlot.yLeft)
		self.curve0.attach(self)

		self.triggerval = 0.0
		self.maxamp0 = 100.0
		self.maxamp1 = 100.0
		self.freeze = 0
		self.average = 0
		self.avcount = 0
		self.logy = 1
		self.datastream = None

		self.dt = 1.0 / samplerate
		self.df = 1.0 / (fftbuffersize * self.dt)
		self.f = np.arange(0.0, samplerate, self.df)
		self.a1 = 0.0 * self.f
		self.a2 = 0.0 * self.f
		
		#print("__init__ self.curve0.setData")
		self.curve0.setData(self.f, self.a1)
		#print("__init__ self.curve1.setData")
		self.curve1.setData(self.f, self.a2)
		
		self.setAxisScale(Qwt.QwtPlot.xBottom, 0.0, 12.5 * initfreq)
		self.setAxisScale(Qwt.QwtPlot.yLeft, -120.0, 0.0)

		self.startTimer(100)
		print("__init__ self.replot")
		self.replot()

	def resetBuffer(self):
		self.df = 1.0 / (fftbuffersize * self.dt)
		self.f = np.arange(0.0, samplerate, self.df)
		self.a1 = 0.0 * self.f
		self.a2 = 0.0 * self.f
		print("FScope.resetBuffer self.curve0.setData")
		self.curve0.setData(self.curve0, self.f, self.a1)
		print("FScope.resetBuffer self.curve1.setData")
		self.curve0.setData(self.curve0, self.f, self.a2)

	def setMaxTime(self, val):
		self.maxtime = val
		self.updateTimer()

	def setTriggerLevel(self, val):
		self.triggerval = val

	def setDisplay(self):
		n = fftbuffersize // 2
		print("n", n)
		if SELECTEDCH == BOTH12:
			#print("FScope.setDisplay self.curve0.setData")
			self.curve0.setData(self.f[:n], self.a1[:n])
			#print("FScope.setDisplay self.curve1.setData")
			self.curve1.setData(self.f[:n], self.a2[:n])
		elif SELECTEDCH == CH2:
			#print("FScope.setDisplay self.curve0.setData")
			self.curve0.setData([0.0, 0.0], [0.0, 0.0])
			#print("FScope.setDisplay self.curve1.setData")
			self.curve1.setData(self.f[:n], self.a2[:n])
		elif SELECTEDCH == CH1:
			#print("FScope.setDisplay self.curve0.setData")
			self.curve0.setData(self.f[:n], self.a1[:n])
			#print("FScope.setDisplay self.curve1.setData")
			self.curve1.setData([0.0, 0.0], [0.0, 0.0])
		#print("FScope.setDisplay self.replot")
		self.replot()

	def getValue(self, index):
		return self.f[index], self.a1[index]

	def setAverage(self, state):
		self.average = state
		self.avcount = 0

	def setFreeze(self, freeze):
		self.freeze = freeze

	def setDatastream(self, datastream):
		self.datastream = datastream

	def timerEvent(self, e):  # FFT
		global fftbuffersize
		if self.datastream is None:
			return
		if self.freeze == 1:
			return
		if SELECTEDCH == BOTH12:
			channel = 12
			X, Y = self.datastream.read(channel, fftbuffersize, verbose)
			if X is None or not len(X):
				return
			data_CH1 = X[:fftbuffersize]
			data_CH2 = Y[:fftbuffersize]
		elif SELECTEDCH == CH1:
			channel = 1
			X = self.datastream.read(channel, fftbuffersize, verbose)
			if X is None or not len(X):
				return
			data_CH1 = X[:fftbuffersize]
			data_CH2 = np.ones((fftbuffersize,))
		elif SELECTEDCH == CH2:
			channel = 2
			X = self.datastream.read(channel, fftbuffersize, verbose)
			if X is None or not len(X):
				return
			data_CH2 = X[:fftbuffersize]
			data_CH1 = np.ones((fftbuffersize,))
		self.df = 1.0 / (fftbuffersize * self.dt)
		self.setAxisTitle(Qwt.QwtPlot.xBottom, "Frequency [Hz] - Bin width %g Hz" % (self.df,))
		self.f = np.arange(0.0, samplerate, self.df)
		if not SPECTRUM_MODULE:
			lenX = fftbuffersize
			window = np.blackman(lenX)
			sumw = np.sum(window * window)
			A = FFT.fft(data_CH1 * window)  # lenX
			B = (A * np.conjugate(A)).real
			A = FFT.fft(data_CH2 * window)  # lenX
			B2 = (A * np.conjugate(A)).real
			sumw *= 2.0  # sym about Nyquist (*4); use rms (/2)
			sumw /= self.dt  # sample rate
			B /= sumw
			B2 /= sumw
		else:
			print("FFT buffer size: %d points" % (fftbuffersize,))
			B = spectrum.Periodogram(np.array(data_CH1, dtype=np.float64), samplerate)
			B.sides = "onesided"
			B.run()
			B = B.get_converted_psd("onesided")
			B2 = spectrum.Periodogram(np.array(data_CH2, dtype=np.float64), samplerate)
			B2.sides = "onesided"
			B2.run()
			B2 = B2.get_converted_psd("onesided")
		if self.logy:
			P1 = np.log10(B) * 10.0
			P2 = np.log10(B2) * 10.0
			P1 -= P1.max()
			P2 -= P2.max()
		else:
			P1 = B
			P2 = B2
		if not self.average:
			self.a1 = P1
			self.a2 = P2
			self.avcount = 0
		else:
			self.avcount += 1
			if self.avcount == 1:
				self.sumP1 = P1
				self.sumP2 = P2
			elif self.sumP1.shape != P1.shape or self.sumP1.shape != P1.shape:
				self.avcount = 1
				self.sumP1 = P1
				self.sumP2 = P2
			else:
				self.sumP1 += P1
				self.sumP2 += P2
			self.a1 = self.sumP1 / self.avcount
			self.a2 = self.sumP2 / self.avcount
		self.setDisplay()


initfreq = 100.0


class FScopeFrame(Qt.QFrame):
	"""
	Power spectrum widget --- contains controls + display
	"""

	def __init__(self, *args):
		super().__init__(*args)
		vknobpos = scopeheight + 30
		hknobpos = scopewidth + 10
		# the following: setPal..  doesn't seem to work on Ein
		try:
			self.setPaletteBackgroundColor(QColor(240, 240, 245))
		except BaseException:
			pass
		self.setFixedSize(scopewidth + 160, scopeheight + 160)
		self.freezeState = 0

		self.knbSignal = LblKnob(self, 160, vknobpos, "Signal", 1)
		self.knbTime = LblKnob(self, 310, vknobpos, "Frequency", 1)
		self.knbTime.setRange(0.1, 1250.0)

		self.knbSignal.setRange(100, 1000000)

		self.plot = FScope(self)
		self.plot.setGeometry(12.5, 10, scopewidth + 120, scopeheight)
		# self.picker = Qwt.QwtPlotPicker(
		#    Qwt.QwtPlot.xBottom,
		#    Qwt.QwtPlot.yLeft,
		#    Qwt.QwtPicker.PointSelection | Qwt.QwtPicker.DragSelection,
		#    Qwt.QwtPlotPicker.CrossRubberBand,
		#    Qwt.QwtPicker.ActiveOnly, #AlwaysOn,
		#    self.plot.canvas())
		# self.picker.setRubberBandPen(Qt.QPen(Qt.Qt.green))
		# self.picker.setTrackerPen(Qt.QPen(Qt.Qt.cyan))

		self.knbTime.knob.valueChanged.connect(self.setTimebase)
		self.knbTime.setValue(1000.0)
		self.knbSignal.knob.valueChanged.connect(self.setAmplitude)
		self.knbSignal.setValue(1000000)

		self.plot.show()

	def _calcKnobVal(self, val):
		ival = np.floor(val)
		frac = val - ival
		if frac >= 0.9:
			frac = 1.0
		elif frac >= 0.66:
			frac = np.log10(5.0)
		elif frac >= np.log10(2.0):
			frac = np.log10(2.0)
		else:
			frac = 0.0
		dt = 10 ** frac * 10 ** ival
		return dt

	def setTimebase(self, val):
		dt = self._calcKnobVal(val)
		self.plot.setAxisScale(Qwt.QwtPlot.xBottom, 0.0, 12.5 * dt)
		#print("FScopeFrame.setTimebase replotting")
		self.plot.replot()

	def setAmplitude(self, val):
		minp = self._calcKnobVal(val)
		self.plot.setAxisScale(Qwt.QwtPlot.yLeft, -int(np.log10(minp) * 20), 0.0)
		#print("FScopeFrame.setAmplitude replotting")
		self.plot.replot()

	def setDatastream(self, datastream):
		self.datastream = datastream

# ---------------------------------------------------------------------


class FScopeDemo(Qt.QMainWindow):
	"""
	Application container  widget

	Contains scope and power spectrum analyser in tabbed windows.
	Enables switching between the two.
	Handles toolbar and status.
	"""

	def __init__(self, *args):
		super().__init__(*args)

		self.freezeState = 0
		self.changeState = 0
		self.averageState = 0
		self.autocState = 0

		self.scope = ScopeFrame(self)
		self.current = self.scope
		self.pwspec = FScopeFrame(self)
		self.pwspec.hide()

		self.stack = Qt.QTabWidget(self)
		self.stack.addTab(self.scope, "scope")
		self.stack.addTab(self.pwspec, "fft")
		self.setCentralWidget(self.stack)

		self.toolBar = Qt.QToolBar(self)
		self.addToolBar(self.toolBar)
		sb = self.statusBar()
		sbfont = Qt.QFont("Helvetica", 12)
		sb.setFont(sbfont)

		self.btnFreeze = Qt.QToolButton(self.toolBar)
		self.btnFreeze.setText("Freeze")
		self.btnFreeze.setIcon(Qt.QIcon(Qt.QPixmap(icons.stopicon)))
		self.btnFreeze.setCheckable(True)
		self.btnFreeze.setToolButtonStyle(Qt.Qt.ToolButtonTextUnderIcon)
		self.toolBar.addWidget(self.btnFreeze)

		self.btnSave = Qt.QToolButton(self.toolBar)
		self.btnSave.setText("Save CSV")
		self.btnSave.setIcon(Qt.QIcon(Qt.QPixmap(icons.save)))
		self.btnSave.setToolButtonStyle(Qt.Qt.ToolButtonTextUnderIcon)
		self.toolBar.addWidget(self.btnSave)

		self.btnPDF = Qt.QToolButton(self.toolBar)
		self.btnPDF.setText("Export PDF")
		self.btnPDF.setIcon(Qt.QIcon(Qt.QPixmap(icons.pdf)))
		self.btnPDF.setToolButtonStyle(Qt.Qt.ToolButtonTextUnderIcon)
		self.toolBar.addWidget(self.btnPDF)

		self.btnPrint = Qt.QToolButton(self.toolBar)
		self.btnPrint.setText("Print")
		self.btnPrint.setIcon(Qt.QIcon(Qt.QPixmap(icons.print_xpm)))
		self.btnPrint.setToolButtonStyle(Qt.Qt.ToolButtonTextUnderIcon)
		self.toolBar.addWidget(self.btnPrint)

		self.btnMode = Qt.QToolButton(self.toolBar)
		self.btnMode.setText("fft")
		self.btnMode.setIcon(Qt.QIcon(Qt.QPixmap(icons.pwspec)))
		self.btnMode.setCheckable(True)
		self.btnMode.setToolButtonStyle(Qt.Qt.ToolButtonTextUnderIcon)
		self.toolBar.addWidget(self.btnMode)

		self.btnAvge = Qt.QToolButton(self.toolBar)
		self.btnAvge.setText("average")
		self.btnAvge.setIcon(Qt.QIcon(Qt.QPixmap(icons.avge)))
		self.btnAvge.setCheckable(True)
		self.btnAvge.setToolButtonStyle(Qt.Qt.ToolButtonTextUnderIcon)
		self.toolBar.addWidget(self.btnAvge)

		self.btnAutoc = Qt.QToolButton(self.toolBar)
		self.btnAutoc.setText("autocorrelation")
		self.btnAutoc.setIcon(Qt.QIcon(Qt.QPixmap(icons.avge)))
		self.btnAutoc.setCheckable(True)
		self.btnAutoc.setToolButtonStyle(Qt.Qt.ToolButtonTextUnderIcon)
		self.toolBar.addWidget(self.btnAutoc)

		# self.lstLabl = Qt.QLabel("Buffer:",self.toolBar)
		# self.toolBar.addWidget(self.lstLabl)
		# self.lstChan = Qt.QComboBox(self.toolBar)
		# self.lstChan.insertItem(0,"8192")
		# self.lstChan.insertItem(1,"16k")
		# self.lstChan.insertItem(2,"32k")
		# self.toolBar.addWidget(self.lstChan)

		self.lstLR = Qt.QLabel("Channels:", self.toolBar)
		self.toolBar.addWidget(self.lstLR)

		self.btnPrint.clicked.connect(self.printPlot)
		self.btnSave.clicked.connect(self.saveData)
		self.btnPDF.clicked.connect(self.printPDF)
		self.btnFreeze.toggled.connect(self.freeze)
		self.btnMode.toggled.connect(self.mode)
		self.btnAvge.toggled.connect(self.average)
		self.btnAutoc.toggled.connect(self.autocorrelation)
		# self.lstChan.activated.connect(self.fftsize)
		
		self.selectorWidgets = []
		for i in range(3): # dimensions
			selectorWgt = Qt.QComboBox(self.toolBar)
			self.selectorWidgets.append(selectorWgt)
			selectorWgt.activated.connect(partial(self.setChannelEvent, i))
			self.toolBar.addWidget(selectorWgt)
		
		#self.scope.picker.moved.connect(self.moved)
		#self.scope.picker.appended.connect(self.appended)
		#self.pwspec.picker.moved.connect(self.moved)
		#self.pwspec.picker.appended.connect(self.appended)
		self.stack.currentChanged.connect(self.mode)
		self.showInfo(cursorInfo)
		#self.showFullScreen()
		#print(self.size())
		print("FScopeDemo inited")

	def showInfo(self, text):
		self.statusBar().showMessage(text)

	def printPlot(self):
		printer = Qt.QPrinter(Qt.QPrinter.HighResolution)

		printer.setOutputFileName("scope-plot.ps")

		printer.setCreator("Ethernet Scope")
		printer.setOrientation(Qt.QPrinter.Landscape)
		printer.setColorMode(Qt.QPrinter.Color)

		docName = self.current.plot.title().text()
		if not docName.isEmpty():
			docName.replace(Qt.QRegExp(Qt.QString.fromLatin1("\n")), self.tr(" -- "))
			printer.setDocName(docName)

		dialog = Qt.QPrintDialog(printer)
		if dialog.exec_():
#            filter = Qwt.PrintFilter()
#            if (Qt.QPrinter.GrayScale == printer.colorMode()):
#                filter.setOptions(
#                    Qwt.QwtPlotPrintFilter.PrintAll
#                    & ~Qwt.QwtPlotPrintFilter.PrintBackground
#                    | Qwt.QwtPlotPrintFilter.PrintFrameWithScales)
			self.current.plot.print_(printer)
		#p = Qt.QPrinter()
		#if p.setup():
		#    self.current.plot.printPlot(p)#, Qwt.QwtFltrDim(200));

	def printPDF(self):
		fileName = Qt.QFileDialog.getSaveFileName(self, "Export File Name", "", "PDF Documents (*.pdf)")

		if not fileName.isEmpty():
			printer = Qt.QPrinter()
			printer.setOutputFormat(Qt.QPrinter.PdfFormat)
			printer.setOrientation(Qt.QPrinter.Landscape)
			printer.setOutputFileName(fileName)

			printer.setCreator("Ethernet Scope")
			self.current.plot.print_(printer)

#        p = QPrinter()
#        if p.setup():
#           self.current.plot.printPlot(p)#, Qwt.QwtFltrDim(200));

	def saveData(self):
		fileName = Qt.QFileDialog.getSaveFileName(self, "Export File Name", "", "CSV Documents (*.csv)")

		if not fileName.isEmpty():
			csvlib.write_csv(fileName,
							 np.vstack((
										np.arange(self.current.plot.a1.shape[0], dtype=int32) / samplerate,
										self.current.plot.a1, 
										self.current.plot.a2)), 
							 ("TIME", "CH1", "CH2"))

	def setChannelEvent(self, idx, itemIdx):
		self.setChannel(idx, self.selectorWidgets[idx].currentData())

	def setChannel(self, idx, item):
		print("channel.item", idx, item)
		self.scope.plot.avcount = 0
		self.pwspec.plot.avcount = 0
		self.scope.plot.curvesSources[0][idx] = item

	def freeze(self, on, changeIcon=True):
		if on:
			self.freezeState = 1
			if changeIcon:
				self.btnFreeze.setText("Run")
				self.btnFreeze.setIcon(Qt.QIcon(Qt.QPixmap(icons.goicon)))
		else:
			self.freezeState = 0
			if changeIcon:
				self.btnFreeze.setText("Freeze")
				self.btnFreeze.setIcon(Qt.QIcon(Qt.QPixmap(icons.stopicon)))
		self.scope.plot.setFreeze(self.freezeState)
		self.pwspec.plot.setFreeze(self.freezeState)

	def average(self, on):
		if on:
			self.averageState = 1
			self.btnAvge.setText("single")
			self.btnAvge.setIcon(Qt.QIcon(Qt.QPixmap(icons.single)))
		else:
			self.averageState = 0
			self.btnAvge.setText("average")
			self.btnAvge.setIcon(Qt.QIcon(Qt.QPixmap(icons.avge)))
		self.scope.plot.setAverage(self.averageState)
		self.pwspec.plot.setAverage(self.averageState)

	def autocorrelation(self, on):
		if on:
			self.autocState = 1
			self.btnAutoc.setText("normal")
			self.btnAutoc.setIcon(Qt.QIcon(Qt.QPixmap(icons.single)))
		else:
			self.autocState = 0
			self.btnAutoc.setText("autocorrelation")
			self.btnAutoc.setIcon(Qt.QIcon(Qt.QPixmap(icons.avge)))
		self.scope.plot.setAutoc(self.autocState)

	def mode(self, on):
		if on:
			self.changeState = 1
			self.current = self.pwspec
			self.btnMode.setText("scope")
			self.btnMode.setIcon(Qt.QIcon(Qt.QPixmap(icons.scope)))
			self.btnMode.setChecked(True)
		else:
			self.changeState = 0
			self.current = self.scope
			self.btnMode.setText("fft")
			self.btnMode.setIcon(Qt.QIcon(Qt.QPixmap(icons.pwspec)))
			self.btnMode.setChecked(False)
		if self.changeState == 1:
			self.stack.setCurrentIndex(self.changeState)
			self.scope.plot.setDatastream(None)
			#print("self.scope.plot.setDatastream", stream)
			self.pwspec.plot.setDatastream(stream)
		else:
			self.stack.setCurrentIndex(self.changeState)
			self.pwspec.plot.setDatastream(None)
			#print("self.scope.plot.setDatastream", stream)
			self.scope.plot.setDatastream(stream)

	def moved(self, e):
		if self.changeState == 1:
			name = "Freq"
		else:
			name = "Time"
		frequency = self.current.plot.invTransform(Qwt.QwtPlot.xBottom, e.x())
		amplitude = self.current.plot.invTransform(Qwt.QwtPlot.yLeft, e.y())
		if name == "Time":
			df = self.scope.plot.dt
			i = int(frequency / df)
			ampa = self.scope.plot.a1[i]
			ampb = self.scope.plot.a2[i]
		else:
			df = self.pwspec.plot.df
			i = int(frequency / df)
			ampa = self.pwspec.plot.a1[i]
			ampb = self.pwspec.plot.a2[i]
		self.showInfo("%s=%g, cursor=%g, A=%g, B=%g" % (name, frequency, amplitude, ampa, ampb))

	def appended(self, e):
		print("s")
		# Python semantics: self.pos = e.pos() does not work; force a copy
		self.xpos = e.x()
		self.ypos = e.y()
		self.moved(e)  # fake a mouse move to show the cursor position

	def setDatastream(self, ds):
		self.scope.setDatastream(ds)
		for selector in self.selectorWidgets:
			print("ds.channels", ds.channels)
			for i, c in enumerate(ds.channels):
				selector.insertItem(i, c.name, c)
		
		for i in range(2):
			self.setChannel(i, ds.channels[i])

def load_cfg():
	default = ".rigol"  # default probe
	conf_path = os.path.expanduser("~/.dualscope123")
	conf = configparser.ConfigParser()
	print("Loaded config file %s" % (conf_path,))
	if not os.path.isfile(conf_path):
		conf.add_section("probes")
		conf.set("probes", "probe", "audio")
		conf.set("DEFAULT", "verbose", "false")
		with open(conf_path, "w") as fp:
			conf.write(fp)
		return load_cfg()
	else:
		conf.read([conf_path])
		if "probes" not in conf.sections() or "DEFAULT" in conf.sections():
			raise configparser.NoSectionError("Malformed config file.")
		try:
			probe_name = conf.get("probes", "probe").strip("\"'").strip()
		except configparser.NoOptionError:
			probe = default[1:]
		try:
			verbose = conf.get("DEFAULT", "verbose").strip("\"'").strip()
		except configparser.NoOptionError:
			verbose = False
	try:
		probe_module = importlib.import_module("." + probe_name, "dualscope123.probes")
	except ImportError:
		probe_module = importlib.import_module(default, "dualscope123.probes")
		probe_name = default[1:]
	if verbose in ("true", "True", "1", "on", "yes", "YES", "Yes", "On"):
		#print("Loaded probe %s" % probe_name)
		verbose = True
	else:
		verbose = False
	return probe_module, verbose


def main():
	global verbose, samplerate, CHUNK, fftbuffersize, stream
	probe, verbose = load_cfg()
	stream = probe.Probe()
	stream.open()
	samplerate = stream.RATE
	CHUNK = stream.CHUNK
	fftbuffersize = CHUNK
	print("fftbuffersize set CHUNK", fftbuffersize)

	app = Qt.QApplication(sys.argv)
	demo = FScopeDemo()
	demo.setDatastream(stream)
	demo.show()

	print("app.exec_()")
	app.exec_()
	stream.close()


if __name__ == "__main__":
	main()
