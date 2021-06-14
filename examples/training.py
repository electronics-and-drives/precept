import torch
from pytorch_lightning import Trainer
from precept import PreceptModule, PreceptDataModule

device_name = "ptmn90"
data_path = "/tmp/precept/data/90nm_bulk_nmos.h5"
model_path = "/tmp/precept/example"

params_x = [ "W", "L", "Vgs", "Vds", "Vbs" ]
params_y = [ "vdsat", "id", "gm", "gds", "fug" ]

num_x = len(params_x)
num_y = len(params_y)

trafo_mask_x = []
trafo_mask_y = [ "id", "gm", "gds", "fug" ]

lambdas_x = []
lambdas_y = [ 0.2, 0.2, 0.2, 0.2 ]

batch_size = 2000
test_split = 0.2
num_workers = 6
rng_seed = 666

data = PreceptDataModule( data_path
                        , params_x, params_y 
                        , trafo_mask_x, trafo_mask_y
                        , lambdas_x, lambdas_y 
                        , batch_size = batch_size
                        , test_split = test_split
                        , num_workers = num_workers
                        , rng_seed = rng_seed
                        , )

model = PreceptModule( num_x, num_y 
                     , model_path = model_path
                     , )

trainer = Trainer( gpus = 1
                 , max_epochs = 1
                 , max_steps = 5
                 , flush_logs_every_n_steps = 1
                 , precision = 64
                 , checkpoint_callback = True
                 , default_root_dir = model_path
                 , )

trainer.fit(model, data)

trainer.save_checkpoint(f"{model_path}/example.ckpt")

script = model.to_torchscript( method = "trace"
                             , example_inputs = torch.rand(1, num_x, dtype=float)
                             , )

torch.jit.save(script, f"{model_path}/example.ckpt")
