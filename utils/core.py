from os.path import join
from ctypes import CDLL

dynamic_lib = CDLL(join('utils', 'physics.so'))
