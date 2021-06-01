import sys
import argparse
import datetime
import yaml
import warnings

from flask import Flask, request, abort

from .mod import PreceptModule
from .dat import PreceptDataModule
from .cli import PreceptCLI
from .srv import PreceptSRV

GPL_NOTICE = f"""
PRECEPT Copyright (C) 2021 Electronics & Drives
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions.
"""

def pct():
    print(GPL_NOTICE)
    cli = PreceptCLI(PreceptModule, PreceptDataModule)
    return 0

def prc():
    srv = PreceptSRV()
    host, port = srv.setup()

    app = Flask("__main__")

    @app.route('/predict', methods=['POST'])
    def predict():
        res = srv.predict(request.json)

        if type(res) is int:
            abort(res)
        elif res is None:
            abort(400)
        else:
            return res

    return app.run(host = host, port = port)

def main():
    print(GPL_NOTICE)
    return 0

if __name__ == '__main__':
    sys.exit(main())
