import yaml
import torch
import pandas as pd
import numpy as np
from precept import PreceptModule, PreceptApproximator

# File System setup
checkpoint_path = '/tmp/precept/example/example.ckpt'
meta_file = '/tmp/precept/example/example.yml'

# Load Model Meta Data
with open(meta_file, 'r') as yml:
    config = yaml.load(yml, Loader = yaml.FullLoader)

# Load Model Checkpoint
model = PreceptModule.load_from_checkpoint(checkpoint_path)

# Define Inference Object
device = PreceptApproximator( model
                            , config['params_x'], config['params_y']
                            , config['num_x'], config['num_y']
                            , config['mask_x'], config['mask_y']
                            , config['min_x'], config['min_y'] 
                            , config['max_x'], config['max_y']
                            , config['lambdas_x'], config['lambdas_y']
                            , )

# Generate some Inputs
X = pd.DataFrame({ x: np.random.rand(5).tolist() for x in config['params_x']})

# Predict corresponding Outputs
Y = device.predict(X)
