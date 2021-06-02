(import [numpy :as np])
(import [scipy :as sp])
(import [pandas :as pd])

(import torch)

(import [pathlib [Path]])

(import [.mod [PreceptModule]])
(import [.utl [bct cbt scl]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(defclass PreceptApproximator []
  (defn __init__ [self model 
                  params-x params-y 
                  num-x num-y
                  mask-x mask-y
                  min-x min-y 
                  max-x max-y
                  lambdas-x lambdas-y]

    (setv self.model model)

    (setv self.params-x   params-x
          self.params-y   params-y
          self.num-x      num-x
          self.num-y      num-y
          self.lambdas-x  lambdas-x
          self.lambdas-y  lambdas-y)

    (setv self.mask-x mask-x
          self.mask-y mask-y
          self.trafo-mask-x (list (map (fn [bcm] 
                                          (in bcm self.mask-x)) 
                                       self.params-x))
          self.trafo-mask-y (list (map (fn [bcm] 
                                          (in bcm self.mask-y)) 
                                       self.params-y)))
    
    (setv self.min-x min-x
          self.min-y min-y
          self.max-x max-x
          self.max-y max-y))

  (defn predict [self inputs]
    (with [(.no-grad torch)]
      (let [X (-> inputs (get self.params-x) 
                         (.to-numpy))

            _ (when self.mask-x
                (setv (get X.T self.trafo-mask-x)
                      (lfor (, idx x) (enumerate (get X.T self.trafo-mask-x))
                        (bct (np.array x) (get self.lambdas-x idx)))))
            
            X′ (. (np.array (lfor (, idx x) (enumerate X.T)
                                  (scl x :min-x (get self.min-x idx) 
                                         :max-x (get self.max-x idx))))
                  T)
            
            Y′ (->> X′ (torch.Tensor)
                       (self.model)
                       (.numpy))
            
            Y (. (np.array (lfor (, idx y) (enumerate Y′.T)
                                 (scl y :min-x (get self.min-y idx) 
                                        :max-x (get self.max-y idx))))
                 T)
            
            _ (when self.mask-y
                (setv (get Y.T self.trafo-mask-y)
                      (lfor (, idx y) (enumerate (get Y.T self.trafo-mask-y))
                        (cbt (np.array y) (get self.lambdas-y idx))))) ]

        (pd.DataFrame Y :columns self.params-y)))))
