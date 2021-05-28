# PRECEPT

**Pr**imitive d**e**vi**ce** a**p**proxima**t**ion, a machine learning
extension for the PREDICT Toolbox based on
[Lightning](https://www.pytorchlightning.ai/). Train Neural
networks with PREDICT data to model the behavior of primitive devices.

## Quick Start

Adjust the `data_path` and `device_name` in `examples/config.yml` and run:

```bash
$ pct --config ./examples/config.yml
```

If you need data, check out 
[pyrdict](https://githhub.com/augustunderground/pyrdict).

## Setup

### Dependencies and Requirements

- `conda`: 4.9.2
- `pip`: 21.0.1
- `python`: 3.8.8
- `CUDA`: 11.2
- `Nvidia Driver`: 460.73.01

Everything else is specified in `requirements.txt`.

### Installation

Clone this repository:

```bash
$ git clone https://github.com/electronics-and-drives/precept.git
```

`cd precept` into the directory and install the package:

```bash
$ pip install .
```

With this the `precept` package including the `pct` command will be available.

## CLI

`precept` comes with a CLI for training based on a `.yml` configuration file.

### Training

```bash
$ pct --config ./examples/config.yml
```

See comments in `./examples/config.yml`

#### Data

The data specified by the `data_path` field is expected to be an HDF and structured like so:

```python
In  [1]: list(map(lambda k: f"{k}: {hdf_file[k].shape}", hdf_file.keys()))
Out [1]: ['columns: (18,)', 'data: (18, 16105100)']
```

Where `columns` are the headers for what is stored in `data`. These **must**
align with the `params_x` and `params_y` specification in the given
`config.yml`.

If you need some toy data, check out 
[pyrdict](https://githhub.com/augustunderground/pyrdict).

### Inference

soon<sup>TM</sup>

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
    - [ ] Implement model inference based on Flask
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
