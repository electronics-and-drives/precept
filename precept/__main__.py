import sys
import argparse
import datetime
import yaml
import warnings

#from .core import *
from .mod import PreceptModule
from .dat import PreceptDataModule
from .cli import PreceptCLI

GPL_NOTICE="""
pct  Copyright (C) 2021 Electronics & Drives
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.
"""

def pct():
    print(GPL_NOTICE)
    cli = PreceptCLI(PreceptModule, PreceptDataModule)
    return 0

def prc():
    print("FOOBARBARBBRBABRA")
    return 0

def main():
    print(GPL_NOTICE)
    return 0

if __name__ == '__main__':
    sys.exit(main())
