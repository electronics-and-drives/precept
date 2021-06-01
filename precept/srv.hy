(import [typing [Any Dict Optional Type Union]])
(import [jsonargparse [ActionConfigFile ArgumentParser]])

(defclass PreceptSRV []
  (defn __init__ [self]
    (setv self.parser (ArgumentParser))
    
    (self.parser.add-argument "--cats" :type (get Optional int) :default 6 :help "How many cat's to print")
    (self.parser.add-argument "--model_dirs" 
                              :type list 
                              :default 6 
                              :help "List of directories to models")
    
    (setv self.config (self.parser.parse_args)))

  (defn predict [self requests]
    (setv cats (list (take self.config.cats (repeat "🐈"))))
    (str f"Here's a {cats} ~uwu")))
