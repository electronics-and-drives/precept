(import torch)
(import [torch.nn :as nn])
(import [torch.optim :as optim])

(import [pytorch-lightning [LightningModule]])
(import [pytorch-lightning.callbacks [ModelCheckpoint]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])

(defclass PreceptModule [LightningModule]
  (defn __init__ [self ^int num-x ^int num-y
                  &optional ^str   [model-path "/tmp/precept"]
                            ^float [learning-rate 0.001]
                            ^float [beta-1 0.9]
                            ^float [beta-2 0.999]]

    f"Operating Point Data Regressor Model

    Mandatory Model Args:
      num_x: Number of input neurons
      num_y: Number of output neurons

    Optional Optimizer Args:
      learning_rate: Learning rate of the ADAM optimizer (default = 0.001)
      beta_1: decay rate (default = 0.9)
      beta_2: decay rate (default = 0.999)
     "

    (.__init__ (super))
    
    (self.save-hyperparameters)

    (setv self.model-path model-path)

    (setv self.learning-rate learning-rate
          self.betas         (, beta-1 beta-2))

    (setv self.cb-checkpoint (ModelCheckpoint :monitor "valid_loss"
                                              :dirpath self.hparams.model-path
                                              :filename "op-{epoch:02d}-{valid_loss:.3f}"
                                              :mode "min"))

    (setv self.net (nn.Sequential (nn.Linear self.hparams.num-x 128)
                                  (nn.ReLU)
                                  (nn.Linear 128 256)
                                  (nn.ReLU)
                                  (nn.Linear 256 512)
                                  (nn.ReLU)
                                  (nn.Linear 512 1024)
                                  (nn.ReLU)
                                  (nn.Linear 1024 512)
                                  (nn.ReLU)
                                  (nn.Linear 512 256)
                                  (nn.ReLU)
                                  (nn.Linear 256 128)
                                  (nn.ReLU)
                                  (nn.Linear 128 self.hparams.num-y)
                                  (nn.ReLU))))

  (defn forward [self x]
    (self.net x))

  (defn training-step [self batch batch-idx]
    (let [(, x y) batch
          y-prime (self.net x)
          train_loss (nn.functional.mse-loss y-prime y)]
      (self.log "train_loss" train_loss :on-step False :on-epoch True 
                                        :prog-bar False :logger True)
      train_loss))

  (defn validation-step [self batch batch-idx]
    (let [(, x y) batch
          y-prime (self.net x)
          valid_loss (nn.functional.l1-loss y-prime y)]
      (self.log "valid_loss" valid_loss :on-step False :on-epoch True 
                                        :prog-bar True :logger True)
      valid_loss))

  (defn configure-optimizers [self]
    (.Adam optim (.parameters self.net) 
                 :lr self.learning-rate 
                 :betas self.betas))

  (defn configure-callbacks [self]
    [self.cb-checkpoint]))
