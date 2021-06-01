# PRECEPT

**Pr**imitive d**e**vi**ce** a**p**proxima**t**ion, a machine learning
extension for the PREDICT Toolbox based on
[Lightning](https://www.pytorchlightning.ai/). Train Neural
networks with PREDICT data to model the behavior of primitive devices.

## Setup

### Dependencies and Requirements

Tested with:

- `conda`: 4.9.2
- `pip`: 21.0.1
- `python`: 3.8.8
- `CUDA`: 11.2
- `Nvidia Driver`: 460.73.01

Everything else is specified in `requirements.txt`. Other/Higher versions of
these dependencies may work, but are untested.

### Installation

Clone this repository:

```bash
$ git clone https://github.com/electronics-and-drives/precept.git
```

`cd precept` into the directory and install the python package:

```bash
$ pip install .
```

With this the `precept` API as well as the two CLIs `pct` (for training) and
`prc` (for inference) will be available.

## CLI

`precept` comes with CLIs for both training and inference based on a `.yml`
configuration files. For more information about all options see the help option.

```sh
$ pct --help

$ prc --help
```

### Training

```bash
$ pct --config ./examples/train.yml
```

Precept comes with the following options:

```yaml
model:
  learning_rate: <float, default = 0.001>
  beta_1:        <float, default = 0.9>
  beta_2:        <float, default = 0.999>
data:
  data_path:     <string>   # Path to HDF5 database
  params_x:      <[string]> # List of input column names 
  params_y:      <[string]> # List of output column names 
  trafo_mask_x:  <[string]> # List of input paramters that will be transformed
  trafo_mask_y:  <[string]> # List of output paramters that will be transformed
  batch_size:    <int, default = 2000>
  test_split:    <float, default = 0.2>
  num_workers:   <int, default = 6>
  rng_seed:      <int>
serialize:       <bool, default = true>
device_name:     <string> # File name for output
model_prefix:    <string> # Path where to store output
```

A default config can be generated by running:

```sh
$ pct --print_config > default.yml
```

Additional documentation for Lightning specific configuration can be found in their
[documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html).

#### Training Data

The data specified by the `data_path` field is expected to be an HDF and
structured like so:

```python
In  [1]: list(map(lambda k: f"{k}: {hdf_file[k].shape}", hdf_file.keys()))
Out [1]: ['columns: (18,)', 'data: (18, 16105100)']
```

Where `columns` are the headers for what is stored in `data`. These **must**
align with the `params_x` and `params_y` specification in the given
`train.yml`.

If you need some toy data, check out 
[pyrdict](https://github.com/AugustUnderground/pyrdict).

### Inference

The inference interface `prc` is much simpler. It takes as input only a
dictionary to all the models that should be served. 

```yaml
models:
    <model-name>: <string> # Path to <name>-model.bin
    ...
```

Start the [flask](https://flask.palletsprojects.com/) server with the `prc`
command and a configuration like the one shown in `examples/infer.yml`.

```sh
$ prc --config ./examples/infer.yml

 * Serving Flask app 'prc' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Then, models can be evaluated by sending requests with the following structure:

```sh
$ curl -X POST -H "Content-Type: application/json" \
       -d '{"<model-name>": {"<param 1>": [vals...], "<param 2>": [vals...], ... }}' \
       127.0.0.1:5000/predict
```

The values for each parameter **must** be in a list, even if it's just one and
there **must** be the same number of values for each parameter and the
parameters **must** have been specified in the `params_x` previously.

## API

soon<sup>TM</sup>

Examples on how to use the API will be in the `examples/` directory of
this repository. There will be documented scripts as well as notebooks.

```python
import precept as pct
```

## TODO

- [X] Split classes into separate modules
- [X] Install requirements in `setup`
- [X] Dump scaler and transformer in `after-fit`
- [X] Infer input and output size from x-parameters and y-parameters
- [X] Implement serialization and compile model trace to torch script
    - [X] Implement model inference based on Flask
    - [ ] Implement model inference CLI in C++ based on torch script models
- [X] CSV Data support
- [ ] TSV, ASCII, PSF, nutmeg, nutbin etc... support
- [ ] Add training and inference API examples
    - [ ] Notebooks as well
- [ ] Add better logging
- [ ] Add manpages for CLI and API

## License

Copyright (C) 2021, Electronics & Drives Lab

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see 
[https://www.gnu.org/licenses/](https://www.gnu.org/licenses).
