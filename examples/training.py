import yaml
import torch
from pytorch_lightning import Trainer
from precept import PreceptModule, PreceptDataModule

# File System setup
device_name = 'example-device'
data_path = '/tmp/precept/data/90nm_bulk_nmos.h5'
model_path = '/tmp/precept/example'

# Specify which columns in the data base are inputs and outputs
params_x = [ 'W', 'L', 'Vgs', 'Vds', 'Vbs' ]
params_y = [ 'vdsat', 'id', 'gm', 'gds', 'fug' ]

# Infer number if input and output neurons
num_x = len(params_x)
num_y = len(params_y)

# Specify which parameters need to be transformed
trafo_mask_x = []
trafo_mask_y = [ 'id', 'gm', 'gds', 'fug' ]

# Specify the transformation paramter
lambdas_x = []
lambdas_y = [ 0.2, 0.2, 0.2, 0.2 ]

# Training Parameters
batch_size = 2000
test_split = 0.2
num_workers = 6
rng_seed = 666

# Data Module Definition
data = PreceptDataModule( data_path                     # Where the data is located
                        , params_x, params_y            # Input and output columns
                        , trafo_mask_x, trafo_mask_y    # Transformation masks
                        , lambdas_x, lambdas_y          # Transformation parameters
                        , batch_size = batch_size       # Batch Size for training
                        , test_split = test_split       # How large the test split is
                        , num_workers = num_workers     # Number of CPU cores for data loading
                        , rng_seed = rng_seed           # Random seed
                        , )

# Model Definition
model = PreceptModule( num_x, num_y                     # Number of inputs and outputs
                     , model_path = model_path          # Where the model will be saved
                     , )

trainer = Trainer( gpus = 1                             # Number of available GPUs
                 , max_epochs = 1                       # Maximum number of epochs
                 , max_steps = 5                        # Maximum number of steps
                 , flush_logs_every_n_steps = 1         # Log frequency
                 , precision = 64                       # Floating point precision
                 , checkpoint_callback = True           # Allow checkpoint for auto saving
                 , default_root_dir = model_path        # Where the models will be saved
                 , )

# Fit the Model
trainer.fit(model, data)

# Save a checkpoint of the current model
trainer.save_checkpoint(f'{model_path}/{device_name}.ckpt')

# Save Metadata for Inference
meta_file = f'{model_path}/{device_name}.yml'

meta_data = { 'num_x':      num_x
            , 'num_y':      num_y
            , 'params_x':   params_x
            , 'params_y':   params_y
            , 'mask_x':     trafo_mask_x
            , 'mask_y':     trafo_mask_y
            , 'min_x':      data.min_x.tolist()
            , 'max_x':      data.max_x.tolist()
            , 'min_y':      data.min_y.tolist()
            , 'max_y':      data.max_y.tolist()
            , 'lambdas_x':  data.lambdas_x
            , 'lambdas_y':  data.lambdas_y
            , }

yaml.Dumper.ignore_aliases = lambda *args: True
with open(meta_file, 'w+') as yml:
    yaml.dump( meta_data, yml
             , allow_unicode = True
             , default_flow_style = False
             , )

# Compile current model to TorchScript
script = model.to_torchscript( method = 'trace'
                             , example_inputs = torch.rand(1, num_x, dtype=float)
                             , )

torch.jit.save(script, f'{model_path}/{device_name}.pt')
