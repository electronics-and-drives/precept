(import dill)
(import [pandas :as pd])
(import [typing [Any Dict Optional Type Union]])
(import [jsonargparse [ArgumentParser ActionConfigFile]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(import [.inf [PreceptApproximator]])

(defclass PreceptSRV []
  (defn __init__ [self]
    (setv self.parser (ArgumentParser))

    (self.parser.add_argument "--host" 
                              :type str
                              :default "localhost" 
                              :help "Host address of flask server.")
    
    (self.parser.add_argument "--port" 
                              :type int
                              :default "5000" 
                              :help "Port of flask server.")
    
    (self.parser.add_argument "--config" :action ActionConfigFile)
    
    (self.parser.add-argument "--models" 
                              :type dict
                              :default {}
                              :help "List of directories to models")
    
    (setv self.args (self.parser.parse_args)))

  (defn setup [self]
    (setv self.models 
      (let [models (vars self.args.models)]
        (dfor mid models
          [ mid 
            (with [file (open (get models mid) "rb")]
              (-> file 
                  (dill.load)
                  (unpack-mapping)
                  (PreceptApproximator))) ])))

    (, self.args.host self.args.port))

  (defn predict [self ^dict inputs]
    (let [model-ids (filter (fn [m] (in m (.keys self.models))) (.keys inputs))]
      (dfor mid model-ids
        [ mid
          (try 
            (setv df (-> inputs (get mid) (pd.DataFrame)))
          (except [e ValueError]
            (print f"ValueError: {e}\nfor inputs: {inputs}\n")
            NaN)
          (else (-> self.models (get mid) (.predict df) (.to-dict)))) ]))))
