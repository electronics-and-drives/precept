(import [typing [Any Dict Optional Type Union]])
(import [jsonargparse [ArgumentParser ActionConfigFile]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])

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
    (print self.args.models)
    ;(let [dirs self.config.model_dirs]
    ;  (lfor d dirs
    ;    (print d)))
    ;(with [dill-file (open self.config "model_dirs")])
  ))
