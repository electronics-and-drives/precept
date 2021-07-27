(import [datetime [datetime]])
(import [pathlib [Path]])

(import yaml)
(import torch)
(import [numpy :as np])
(import [pandas :as pd])

(import [pytorch-lightning.utilities.cli [LightningCLI]])
(import [jsonargparse.typing [Path-fr Path-dw]])

(import [.mod [PreceptModule]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])

(setv yaml.Dumper.ignore-aliases (fn [&rest args] True))

(defclass PreceptCLI [LightningCLI]
  (defn add-arguments-to-parser [self parser]
    (let [num-params (fn [ps] (len ps))
          ts-dir (fn [pf dn] 
                  (let[ts (-> datetime
                              (.now) 
                              (.strftime "%Y%m%d-%H%M%S"))]
                    (.format "{}/op-{}-{}" pf dn ts)))]

      (parser.add-argument "--serialize"    :default True)
      (parser.add-argument "--device_name"  :default "mos")
      (parser.add-argument "--model_prefix" :default "/tmp/precept")

      (parser.link-arguments "data.params_x" "model.num_x"
                             :compute-fn num-params)
      (parser.link-arguments "data.params_y" "model.num_y"
                             :compute-fn num-params)

      (parser.link-arguments ["model_prefix" "device_name"] 
                             "trainer.default_root_dir"
                             :compute-fn ts-dir)

      (parser.link-arguments "trainer.default_root_dir" "model.model_path")))

  (defn before-fit [self]
    (let [device-name (get self.config "device_name")
          model-path  (get self.config "trainer" "default_root_dir")]
      (.mkdir (Path model-path) :parents True :exist-ok True)))

  (defn after-fit [self]
    (let [model-path  (get self.config "trainer" "default_root_dir")
          device-name (get self.config "device_name")

          best-path   self.model.cb-checkpoint.best-model-path
          model-ckpt  (PreceptModule.load-from-checkpoint best-path)

          model-file  (.format "{}/{}-model.yml" model-path device-name)
          model-data  { "num_x"      self.datamodule.num-x
                        "num_y"      self.datamodule.num-y
                        "params_x"   self.datamodule.params-x
                        "params_y"   self.datamodule.params-y
                        "mask_x"     self.datamodule.mask-x
                        "mask_y"     self.datamodule.mask-y
                        "min_x"      (.tolist self.datamodule.min-x)
                        "max_x"      (.tolist self.datamodule.max-x)
                        "min_y"      (.tolist self.datamodule.min-y)
                        "max_y"      (.tolist self.datamodule.max-y)
                        "trafo_type" self.datamodule.trafo-type
                        "lambdas_x"  (list (filter (fn [l] l) self.datamodule.lambdas-x))
                        "lambdas_y"  (list (filter (fn [l] l) self.datamodule.lambdas-y)) } ]

      (.eval model-ckpt)
      (.freeze model-ckpt)

      (with [yml-file (open model-file "w+")]
        (yaml.dump model-data yml-file 
                   ;:default-flow-style False
                   :allow-unicode True ))

      (if (get self.config "serialize")
        (-> model-ckpt 
            (.to-torchscript :method "trace" 
                             :example-inputs (torch.rand 1 (get model-data "num_x")))
            (.save (.format "{}/{}-model.pt" model-path device-name)))
        (-> model-ckpt 
            (.state-dict) 
            (torch.save (.format "{}/{}-model.ckpt" model-path device-name)))))))
