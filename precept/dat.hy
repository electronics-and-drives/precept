(import [numpy :as np])
(import [scipy :as sp])
(import [pandas :as pd])
(import [h5py :as h5])
(import [multiprocess :as mp])

(import [pathlib [Path]])

(import torch)
(import [torch.utils.data [random_split TensorDataset DataLoader]])

(import [pytorch-lightning [LightningDataModule]])

(import [sklearn.preprocessing [ PowerTransformer power-transform 
                                 MinMaxScaler minmax-scale 
                                 MaxAbsScaler maxabs-scale
                                 QuantileTransformer quantile-transform
                                 normalize ]])
(import [sklearn.model_selection._split [train_test_split]])
(import [sklearn.utils [shuffle]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])


(defn _bct [y &optional [λ 0.2]]
f"Box-Cox Transformation
Takes a scalar or 1D np.array and transforms it
according to:

        λ  
       y -1
  y' = ――――   if λ ≠ 0
        λ  

  y' = ln(y)  if λ = 0

Returns the transformed scalar or vector.
"
  (if (= λ 0)
    (np.log y)
    (/ (- (np.power y λ) 1) λ)))

(defn _cbt [y′ &optional [λ 0.2]]
f"Inverse Box-Cox Transformation (Cox-Box)
Takes a scalar or 1D np.array and transforms it
according to:

       ⎛ln(y'∙λ+1)⎞
       ⎜――――――――――⎟
       ⎝    λ     ⎠
  y = e               if λ ≠ 0

       y'
  y = e               if λ = 0

Returns the inverse transformed scalar or vector.
"
  (if (= λ 0)
    (np.exp y′)
    (np.exp (/ (np.log (+ (* y′ λ) 1)) λ))))



(defclass PreceptDataModule [LightningDataModule]
  (defn __init__ [self ^str  data-path 
                       ^list params-x 
                       ^list params-y 
                       ^list trafo-mask-x 
                       ^list trafo-mask-y
                  &optional ^int   [batch-size 2000]
                            ^float [test-split 0.2]
                            ^int   [num-workers (-> mp (.cpu-count) (/ 2) (int) (max 1))]
                            ^int   [rng-seed 666]
                            ^float [sample-ratio 0.75]
                            ^list  [lambdas-x [0.2]]
                            ^list  [lambdas-y [0.2]]]

    f"Precept Operating Point Data Module
    Mandatory Args:
    ---------------
      data_path:    Path to HDF5 database
      params_x:     List of input parameters
      params_y:     List of output parameters
      trafo_mask_x: input parameters that will be transformed
      trafo_mask_y: output parameters that will be transformed

    Optional Args:
    --------------
      batch_size:   default = 2000
      test_split:   split ratio between training and test data (default = 0.2)
      num_workers:  number of cpu cores for loading data (default = 6)
      rng_seed:     seed for random number generator (default = 666)
      sample_ratio: ratio sampled from triode region vs saturation region
                    (default = 0.75)
      lambdas-x/y:  list of lambdas for each parameter nambed in trafo-mask-x/y,
                    or a list with a single value, resulting in the same for all.
    "

    (.__init__ (super))

    (setv self.data-path    data-path
          self.batch-size   batch-size
          self.test-split   test-split

          self.sample-ratio sample-ratio
          self.num-workers  num-workers
          self.rng-seed     rng-seed

          self.params-x     params-x
          self.params-y     params-y
          
          self.num-x        (len self.params-x)
          self.num-y        (len self.params-y))

    ;; Converting the column names based trafo mask to a bit mask
    ;; for accessing a np array instead of a data frame
    (setv self.trafo-mask-x (list (map (fn [mask] (in mask trafo-mask-x)) 
                                       params-x))
          self.trafo-mask-y (list (map (fn [mask] (in mask trafo-mask-y)) 
                                       params-y)))

    ;; Check if mask and lambdas line up, if they don't take the first from
    ;; the lambdas list and repeat it as many times as necessary
    (setv self.lambdas-x    (if (or (= (len trafo-mask-x) (len lambdas-x)) 
                                    (not lambdas-x))
                                lambdas-x 
                                (repeat (first lambdas-x) (len trafo-mask-x)))
          self.lambdas-y    (if (or (= (len trafo-mask-y) (len lambdas-y)) 
                                    (not lambdas-y))
                                lambdas-y 
                                (repeat (first lambdas-y) (len trafo-mask-y))))

    ;(setv self.x-trafo      (QuantileTransformer :random-state self.rng-seed
    ;                                             :output-distribution "normal")
    ;      self.y-trafo      (QuantileTransformer :random-state self.rng-seed
    ;                                             :output-distribution "normal")
    ;      self.x-scaler     (MinMaxScaler :feature-range (, 0 1))
    ;      self.y-scaler     (MinMaxScaler :feature-range (, 0 1)))


    )

  (defn prepare-data [self]
    (let [file-type (. (Path self.data-path) suffix)
          process-df (fn [df]
                      (setv (get df "gmid") (/ (get df "gm")
                                               (get df "id"))
                            (get df "Jd") (/ (get df "id") 
                                          (get df "W"))
                            (get df "A0") (/ (get df "gm") 
                                             (get df "gds")))
                      (.dropna df))]
      (setv self.data-frame 
        (cond [(in file-type [".h5" ".hdf" ".hdf5"])
               (with [hdf-file (h5.File self.data-path "r")]
                (let [column-names (->> "columns" (get hdf-file) 
                                                  (map (fn [c] (.decode c "UTF-8"))) 
                                                  (list))
                      data-matrix (->> "data" (get hdf-file) 
                                              (np.array) 
                                              (np.transpose))
                      df (pd.DataFrame data-matrix :columns column-names)]
                  (process-df df)))]
              [(in file-type [".csv"])
               (-> self.data-path
                   (pd.read-csv)
                   (process-df))]
              [(in file-type [".tsv"])
               (-> self.data-path
                   (pd.read-csv :delim_whitespace True)
                   (process-df))]
              [True (raise (TypeError (.format "File type {} not Supported!" file-type)))]))
      (setv self.dims self.data-frame.shape)))

  (defn setup [self &optional [stage None]]
    (if (or (= stage "fit") (is stage None))
      (let [sat-mask (. (& (>= self.data-frame.Vds (- self.data-frame.Vgs self.data-frame.vth))
                           (> self.data-frame.Vgs self.data-frame.vth))
                      values)

            num-samples (-> self.data-frame (. shape) (first) (/ 4) (int))

            sdf (get self.data-frame sat-mask (slice None))
            sdf-weights (minmax-scale (- (sp.stats.zscore sdf.id.values)))
            sat-samp (.sample sdf :n (int (* num-samples self.sample-ratio))
                                  :weights sdf-weights
                                  :replace False 
                                  :random-state self.rng-seed )

            tdf (get self.data-frame (~ sat-mask) (slice None))
            tdf-weights (minmax-scale (- (sp.stats.zscore tdf.id.values)))
            tri-samp (.sample tdf :n (int (* num-samples (- 1.0 self.sample-ratio)))
                            :weights tdf-weights
                            :replace False 
                            :random-state self.rng-seed )

            df (shuffle (pd.concat [sat-samp tri-samp] :ignore-index True))

            raw-x (.to-numpy (get df self.params-x))
            raw-y (.to-numpy (get df self.params-y))

            ;transform (fn [array mask trafo] 
            ;            (let [masked-array (get array (, (slice None) mask))
            ;                  trafo-array (.fit-transform trafo masked-array)]
            ;              (setv (get array (, (slice None) mask)) trafo-array)
            ;              array))
            ;trafo-x (if (any self.trafo-mask-x)
            ;            (transform raw-x self.trafo-mask-x self.x-trafo)
            ;            raw-x)
            ;trafo-y (if (any self.trafo-mask-y)
            ;            (transform raw-y self.trafo-mask-y self.y-trafo)
            ;            raw-y)

            trafo-x (if (and (any self.trafo-mask-x) self.lambdas-x)
                        (. (np.array (lfor (, l m x)
                                           (zip self.lambdas-x
                                                self.trafo-mask-x
                                                raw-x.T)
                                           (if m (_bct x l) x))) 
                           T)
                        raw-x)

            trafo-y (if (and (any self.trafo-mask-y) self.lambdas-y)
                        (. (np.array (lfor (, l m y) 
                                           (zip self.lambdas-y
                                                self.trafo-mask-y
                                                raw-y.T) 
                                           (if m (_bct y l) y) )) 
                           T)
                        raw-y)

            data-x (.fit-transform self.x-scaler trafo-x)
            data-y (.fit-transform self.y-scaler trafo-y)

            (, train-x
               valid-x
               train-y
               valid-y) (train-test-split data-x
                                          data-y
                                          :test-size self.test-split
                                          :shuffle True
                                          :random-state self.rng-seed)]

        (setv self.train-set (TensorDataset (torch.Tensor train-x)
                                            (torch.Tensor train-y))
              self.valid-set (TensorDataset (torch.Tensor valid-x) 
                                            (torch.Tensor valid-y)))

        (setv self.dims (-> self.train-set (get 0) (get 0) (. shape)))))

    ;(if (or (= stage "test") (is stage None))
    ;  None)
  )

  (defn train-dataloader [self]
    (DataLoader self.train-set :batch-size self.batch-size 
                               :num-workers self.num-workers 
                               :pin-memory True))

  (defn val-dataloader [self]
    (DataLoader self.valid-set :batch-size self.batch-size 
                               :num-workers self.num-workers 
                               :pin-memory True))

  ;(defn test-dataloader [self]
  ;  None)
)
