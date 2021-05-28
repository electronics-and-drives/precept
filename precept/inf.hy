(import [numpy :as np])
(import [scipy :as sp])

(import joblib)
(import torch)

;(import [sklearn.preprocessing [ PowerTransformer power-transform 
;                                 MinMaxScaler minmax-scale 
;                                 MaxAbsScaler maxabs-scale
;                                 QuantileTransformer quantile-transform
;                                 normalize ]])

(import [pathlib [Path]])

;(import [.mod [PreceptModule]])
(import [precept [PreceptModule]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])

(defclass PreceptApproximator []
  (defn __init__ [self ^str model-path
                  &optional [base-dir    None]
                            [base-name   None]
                            [trafo-x-dir None]
                            [trafo-y-dir None]
                            [scale-x-dir None]
                            [scale-y-dir None]]
    (setv self.model-path (Path model-path))
    (when (not (self.model-path.exists))
      (raise (TypeError (.format "Cannot find model file: {}"
                                 (str self.model-path)))))
    (setv self.model 
      (cond [(= ".ckpt" self.model-path.suffix)
             (PreceptModule.load-from-checkpoint model-path)]
            [(= ".pt" self.model-path.suffix)
             (raise (TypeError "Loading torch script models not yet implemented!"))]
            [True
             (raise (TypeError (.format "Model File type {} not Supported!" 
                                        self.model-path.suffix)))]))
    (setv (, self.trafo-x
             self.trafo-y
             self.scale-x
             self.scale-y) (cond [(not (is base-name None))
                                  (lfor ext ["x.trafo" "y.trafo" "x.scale" "y.trafo"] 
                                    (.load joblib (.format "{}/{}-{}" 
                                                           (str (first self.model-path.parents)) 
                                                           base-name ext))) ]
                                 [(all [base-dir base-name])
                                  (lfor ext ["x.trafo" "y.trafo" "x.scale" "y.trafo"] 
                                    (.load joblib (.format "{}/{}-{}" 
                                                           (str (first self.model-path.parents)) 
                                                           base-name ext)))
                                 ]
                                 [(all [trafo-x-dir trafo-y-dir scale-x-dir scale-y-dir])
                                 ]
                                 [True
                                  (raise (TypeError "No transformations given.")) ]))
    (.eval self.model)
    (.freeze self.model))
)

(setv model-path "./models/op-ptmp90-20210528-152333/op-epoch=00-valid_loss=0.018.ckpt")
(setv model-path "./models/op-ptmp90-20210528-152333/ptmp90-model.pt")

(.format "{}/{}-{}" (str (first (. (Path model-path) parents))) "ptmp90" "model.pt")

(setv path (Path model-path))
(str (first path.parents))

(setv apx (PreceptApproximator model-path :base-name "ptmp90"))
(PreceptApproximator "foo.asdf")

