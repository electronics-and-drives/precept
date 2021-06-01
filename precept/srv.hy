(import dill)
(import [pandas :as pd])
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
    
    (setv self.args (self.parser.parse_args))
    (self.init-models))

  (defn init-models [self]
    (setv self.models 
      (let [models (vars self.args.models)]
        (dfor mid models
          [ mid 
            (with [file (open (get models mid) "rb")]
              (-> file 
                  (dill.load)
                  (unpack-mapping)
                  (PreceptApproximator))) ]))))

  (defn predict [self ^dict inputs]
    (try 
      (pd.DataFrame inputs)
      (except [e ValueError]
        (print f"ValueError: {e}\nfor inputs: {inputs}\n")
        f"ValueError: {e}\n")
      (else "~uwu"))))
