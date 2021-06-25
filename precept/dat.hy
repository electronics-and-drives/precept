(import [numpy :as np])
(import [pandas :as pd])
(import [h5py :as h5])
(import [multiprocess :as mp])

(import [pathlib [Path]])

(import torch)
(import [torch.utils.data [random_split TensorDataset DataLoader]])

(import [pytorch-lightning [LightningDataModule]])

(import [.utl [bct cbt scl]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(defclass PreceptDataModule [LightningDataModule]
  (defn __init__ [self ^list params-x 
                       ^list params-y 
                       ^list trafo-mask-x 
                       ^list trafo-mask-y
                       ^list lambdas-x
                       ^list lambdas-y
                  &optional ^int   [batch-size 2000]
                            ^float [test-split 0.2]
                            ^int   [num-workers (-> mp (.cpu-count) (/ 2) (int) (max 1))]
                            ^int   [rng-seed 666]
                            ^bool  [scale True]]

    f"Precept Operating Point Data Module
    Mandatory Args:
      data_path:    Path to HDF5 database
      params_x:     List of input parameters
      params_y:     List of output parameters
      trafo_mask_x: input parameters that will be transformed
      trafo_mask_y: output parameters that will be transformed
      lambdas-x/y:  list of lambdas for each parameter nambed in trafo-mask-x/y,
                    or a list with a single value, resulting in the same for all.

    Optional Args:
      batch_size:   default = 2000
      test_split:   split ratio between training and test data (default = 0.2)
      num_workers:  number of cpu cores for loading data (default = 6)
      rng_seed:     seed for random number generator (default = 666)
      scale:        True (default) if the data should be scaled between [0;1]
    "

    (.__init__ (super))

    (setv self.batch-size   batch-size
          self.test-split   test-split

          self.num-workers  num-workers
          self.rng-seed     rng-seed

          self.params-x     params-x
          self.params-y     params-y
          
          self.num-x        (len self.params-x)
          self.num-y        (len self.params-y)

          self.min-x        (list (repeat -Inf self.num-x))
          self.max-x        (list (repeat Inf self.num-x))
          self.min-y        (list (repeat -Inf self.num-y))
          self.max-y        (list (repeat Inf self.num-y)))

    ;; Converting the column names based trafo mask to a bit mask
    ;; for accessing a np array instead of a data frame
    (setv self.mask-x trafo-mask-x
          self.mask-y trafo-mask-y
          self.trafo-mask-x (list (map (fn [param] (in param trafo-mask-x)) 
                                       params-x))
          self.trafo-mask-y (list (map (fn [param] (in param trafo-mask-y)) 
                                       params-y)))

    (setv self.lambdas-x lambdas-x
          self.lambdas-y lambdas-y)

    (setv self.data-frame (pd.DataFrame)))

  (defn prepare-data [self]
    (setv self.dims self.data-frame.shape))

  (defn setup [self &optional [stage None]]
    (if (or (= stage "fit") (is stage None))
      (let [df (.sample self.data-frame :frac 1)

            raw-x (.to-numpy (get df self.params-x))
            raw-y (.to-numpy (get df self.params-y))

            _ (when (and (any self.trafo-mask-x) self.lambdas-x)
                (setv (get raw-x.T self.trafo-mask-x)
                      (lfor (, idx x) (enumerate (get raw-x.T self.trafo-mask-x))
                        (bct (np.array x) (get self.lambdas-x idx)))))

            _ (when (and (any self.trafo-mask-y) self.lambdas-y)
                (setv (get raw-y.T self.trafo-mask-y)
                      (lfor (, idx y) (enumerate (get raw-y.T self.trafo-mask-y))
                        (bct (np.array y) (get self.lambdas-y idx)))))

            data-x (if scale (np.apply-along-axis scl 0 raw-x) raw-x)
            data-y (if scale (np.apply-along-axis scl 0 raw-y) raw-y)

            num-train-samples (int (* (- 1.0 self.test-split) (first data-x.shape)))
            sample-idx        (np.array (range (first data-x.shape)))

            train-idx (np.random.choice sample-idx 
                                        num-train-samples
                                        :replace False)

            valid-idx (get sample-idx (np.in1d sample-idx 
                                               train-idx 
                                               :assume-unique True 
                                               :invert True))

            train-x (get data-x train-idx)
            train-y (get data-y train-idx)
            valid-x (get data-x valid-idx)
            valid-y (get data-y valid-idx) ]

        (setv self.train-set (TensorDataset (torch.Tensor train-x)
                                            (torch.Tensor train-y))
              self.valid-set (TensorDataset (torch.Tensor valid-x) 
                                            (torch.Tensor valid-y)))

        (setv self.min-x (np.min raw-x 0)
              self.max-x (np.max raw-x 0)
              self.min-y (np.min raw-y 0)
              self.max-y (np.max raw-y 0))

        (setv self.dims (-> self.train-set (get 0) (get 0) (. shape)))))

    (if (or (= stage "test") (is stage None))
      pass))

  (defn train-dataloader [self]
    (DataLoader self.train-set :batch-size self.batch-size 
                               :num-workers self.num-workers 
                               :pin-memory True))

  (defn val-dataloader [self]
    (DataLoader self.valid-set :batch-size self.batch-size 
                               :num-workers self.num-workers 
                               :pin-memory True))

  (defn test-dataloader [self]
    pass))

(defclass PreceptDataFrameModule [PreceptDataModule]
  (defn __init__ [self ^pd.core.frame.DataFrame  data-frame
                       ^list params-x 
                       ^list params-y 
                       ^list trafo-mask-x 
                       ^list trafo-mask-y
                       ^list lambdas-x
                       ^list lambdas-y
                  &optional ^int   [batch-size 2000]
                            ^float [test-split 0.2]
                            ^int   [num-workers (-> mp (.cpu-count) 
                                                       (/ 2) 
                                                       (int) 
                                                       (max 1))]
                            ^int   [rng-seed 666]]
    (.__init__ (super) params-x params-y trafo-mask-x trafo-mask-y 
                       lambdas-x lambdas-y
                       :batch-size batch-size :test-split test-split 
                       :num-workers num-workers :rng-seed rng-seed)
    (setv self.data-frame data-frame)))

(defclass PreceptDataBaseModule [PreceptDataModule]
  (defn __init__ [self ^str  data-path
                       ^list params-x 
                       ^list params-y 
                       ^list trafo-mask-x 
                       ^list trafo-mask-y
                       ^list lambdas-x
                       ^list lambdas-y
                  &optional ^int   [batch-size 2000]
                            ^float [test-split 0.2]
                            ^int   [num-workers (-> mp (.cpu-count) 
                                                       (/ 2) 
                                                       (int) 
                                                       (max 1))]
                            ^int   [rng-seed 666]]
    (.__init__ (super) params-x params-y trafo-mask-x trafo-mask-y 
                       lambdas-x lambdas-y
                       :batch-size batch-size :test-split test-split 
                       :num-workers num-workers :rng-seed rng-seed)
    (setv self.data-path data-path))

  (defn prepare-data [self]
    (setv self.data-frame
          (let [file-type (. (Path self.data-path) suffix)]
            (cond [(in file-type [".h5" ".hdf" ".hdf5"])
                   (with [hdf-file (h5.File self.data-path "r")]
                      (let [ groups (list (.keys hdf-file))
                             column-names (cond [(in "columns" groups)
                                                 (->> "columns" (get hdf-file) 
                                                      (map (fn [c] (.decode c "UTF-8"))) 
                                                      (list)) ]
                                                [(-> (+ self.params-x self.params-y)
                                                     (set) (.issubset groups))
                                                 groups ]
                                                [True
                                                 (raise (TypeError 
                                                    (+ "Couldn't find Group `columns`"
                                                       "or all parameter names." ))) ])
                             data-matrix (-> (cond [(in "columns" groups)
                                                    (get hdf-file "data") ]
                                                   [(-> (+ self.params-x self.params-y)
                                                     (set) (.issubset groups))
                                                    (lfor c column-names (get hdf-file c)) ]
                                                   [True
                                                    (raise (TypeError 
                                                     (+ "Couldn't find Group `data`"
                                                        "or all parameter names." )))])
                                             (np.array)
                                             (np.transpose))
                             df (pd.DataFrame data-matrix :columns column-names)]
                        (.dropna df)))]
                  [(in file-type [".csv"])
                   (-> self.data-path
                       (pd.read-csv)
                       (.dropna df))]
                  [(in file-type [".tsv"])
                   (-> self.data-path
                       (pd.read-csv :delim_whitespace True)
                       (.dropna df))]
                  [True (raise (TypeError (.format "File type {} not Supported!" file-type)))])))
    (setv self.dims self.data-frame.shape)))
