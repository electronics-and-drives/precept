"""
Neural Networks modeling Primitive Devices

Pipeline for training Neural Networks on Operating Point data
for different mappings.
"""

import hy
from .mod import PreceptModule
from .dat import PreceptDataFrameModule, PreceptDataBaseModule
from .inf import PreceptApproximator
