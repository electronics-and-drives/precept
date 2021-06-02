(import yaml)
(import torch)
(import [pandas :as pd])
(import [typing [Any Dict Optional Type Union]])
(import [jsonargparse [ArgumentParser ActionConfigFile]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(import [.inf [PreceptApproximator]])
(import [.mod [PreceptModule]])

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
            (let [config-path (. (get models mid) config_path)
                  model-path (. (get models mid) model_path)

                  config (with [file (open config-path "r")]
                           (yaml.load file :Loader yaml.FullLoader))

                  _ (setv (get config "model")
                          (cond [(.endswith model-path ".pt")
                                 (torch.jit.load model-path)]
                                [(.endswith model-path ".ckpt")
                                 (PreceptModule.load-from-checkpoint model-path) ]
                                [True
                                 (raise (IOError "Wrong filetype, has to be (.ckpt) or (.pt)"))]))]
              (-> config (unpack-mapping) (PreceptApproximator)))])))
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
