from packages.scipandas import DataScan
from packages.readers import read_civa_bscan
from packages import utils
from matplotlib import pyplot
from os.path import join


bscan = DataScan(read_civa_bscan('B04_I_5MHz_45S_+x_QuarterInch_bscan.txt'))
bscan.coords['Z'] *= utils.time_to_depth_factor(45, 3150)
bscan = bscan.operate('nde')
skewed = bscan.skew(45, 'X', interpolate=True, ts=0.1, fill_value=-1, method='linear')

pyplot.figure(figsize=(3, 3))
pyplot.pcolormesh(skewed.X, skewed.Z, skewed)
pyplot.gca().invert_yaxis()
pyplot.gca().set_aspect('equal')
#
# # pyplot.figure()
# # pyplot.pcolormesh(bscan.X, bscan.Z, bscan)
pyplot.show()
# pyplot.tight_layout()
# pyplot.savefig('linear_skewed.png')
#
