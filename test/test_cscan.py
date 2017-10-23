from packages.scipandas import DataScan
from packages.readers import read_civa_bscan, read_ultravision
from packages import utils
from matplotlib import pyplot
from os.path import join
import importlib

wave = read_ultravision(join('packages', 'scipandas', 'test', 'specimen_1-probe_6-skew_180.txt'),
                        45, 100e6, 5750)
wave.coords['Z'] *= utils.time_to_depth_factor(45, 5750)
wave = DataScan(wave)
cs = utils.cscan(wave)

cs.plot()
pyplot.show()
