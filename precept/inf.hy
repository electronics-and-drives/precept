(import [numpy :as np])
(import [scipy :as sp])
(import [pandas :as pd])

(import joblib)
(import torch)

;(import [sklearn.preprocessing [ PowerTransformer power-transform 
;                                 MinMaxScaler minmax-scale 
;                                 MaxAbsScaler maxabs-scale
;                                 QuantileTransformer quantile-transform
;                                 normalize ]])

(import [pathlib [Path]])

(import [.mod [PreceptModule]])
;(import [precept [PreceptModule]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])

(defclass PreceptApproximator []
  (defn __init__ [self model 
                  params-x params-y 
                  num-x num-y
                  scale-x scale-y 
                  mask-x mask-y
                  trafo-x trafo-y]

    (setv self.model model)
    (.eval self.model)
    (.freeze self.model)

    (setv self.params-x params-x
          self.params-y params-y
          self.num-x (len params-x)
          self.num-y (len params-y))

    (setv self.mask-x mask-x
          self.mask-y mask-y
          self.trafo-mask-x (list (map (fn [bcm] 
                                          (in bcm self.mask-x)) 
                                       self.params-x))
          self.trafo-mask-y (list (map (fn [bcm] 
                                          (in bcm self.mask-y)) 
                                       self.params-y)))
    (setv self.x-scaler scale-x
          self.y-scaler scale-y
          self.x-trafo trafo-x
          self.y-trafo trafo-y)
  )

  (defn predict [self inputs]
    (with [(.no-grad torch)]
      (let [ X (-> inputs (get self.params-x) 
                          (.to-numpy))

             _ (when self.mask-x 
                  (setv (get X (, (slice None) self.trafo-mask-x))
                        (self.x-trafo.transform (get X (, (slice None) 
                                                       self.trafo-mask-x)))))

             Y (-> X (self.x-scaler.transform)
                     (torch.DoubleTensor)
                     (self.model)
                     (.numpy)
                     (self.y-scaler.inverse-transform))

             _ (when self.mask-y
                  (setv (get Y (, (slice None) self.trafo-mask-y))
                        (self.y-trafo.transform (get Y (, (slice None) 
                                                       self.trafo-mask-y)))))]
        (pd.DataFrame Y :columns self.params-y)))))

;(setv model-path "./models/op-ptmp90-20210528-152333/op-epoch=00-valid_loss=0.018.ckpt")
;(setv model-path "./models/op-ptmp90-20210528-152333/ptmp90-model.pt")
;
;(.format "{}/{}-{}" (str (first (. (Path model-path) parents))) "ptmp90" "model.pt")
;
;(setv path (Path model-path))
;(str (first path.parents))
;
;(setv apx (PreceptApproximator model-path :base-name "ptmp90"))
;(PreceptApproximator "foo.asdf")

