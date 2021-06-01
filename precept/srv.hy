(import dill)
(import [typing [Any Dict Optional Type Union]])
(import [jsonargparse [ArgumentParser ActionConfigFile]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])

(import [.inf [PreceptApproximator]])

(defclass PreceptSRV []
  (defn __init__ [self]
    (setv self.parser (ArgumentParser))
    
    (self.parser.add_argument "--config" :action ActionConfigFile)
    
    (self.parser.add-argument "--models" 
                              :type dict
                              :default {}
                              :help "List of directories to models")
    
    (setv self.args (self.parser.parse_args)))

  (defn predict [self requests]
    (let [models (vars self.args.models)]
      (lfor mid models
        (let [ model-file (with [dill-file (open (get models mid) "rb")]
                            (dill.load dill-file)) 
             ]
          (print model-file)
        )))))
