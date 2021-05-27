(import [datetime [datetime]])
(import [pathlib [Path]])

(import joblib)
(import torch)

(import [pytorch-lightning.utilities.cli [LightningCLI]])
(import [jsonargparse.typing [Path_fr Path_dw]])

(import [.mod [PreceptModule]])

(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])

(defclass PreceptCLI [LightningCLI]
  (defn add-arguments-to-parser [self parser]
    (let [num-params (fn [ps] (len ps))]
      (parser.add-argument "--serialize"    :default True)
      (parser.add-argument "--device_name"  :default "mos")
      (parser.add-argument "--model_prefix" :default "/tmp/precept")

      (parser.link-arguments "data.params_x" "model.num_x"
                             :compute-fn num-params)
      (parser.link-arguments "data.params_y" "model.num_y"
                             :compute-fn num-params)

      (parser.link-arguments ["model_prefix" "device_name"] "trainer.default_root_dir"
                              :compute-fn (fn [pf dn] 
                                            (let[ts (-> datetime
                                                        (.now) 
                                                        (.strftime "%Y%m%d-%H%M%S"))]
                                              (.format "{}/op-{}-{}" pf dn ts))))

      (parser.link-arguments "trainer.default_root_dir" "model.model_path")))

  (defn before-fit [self]
    (let [device-name (get self.config "device_name")
          model-path  (get self.config "trainer" "default_root_dir")]
      (.mkdir (Path model-path) :parents True :exist-ok True)))

  (defn after-fit [self]
    (let [model-path  (get self.config "trainer" "default_root_dir")
          device-name (get self.config "device_name")
          num-x       (get self.config "model" "num_x")]

      (.dump joblib self.datamodule.x-trafo  (.format "{}/{}-x.trafo" model-path device-name))
      (.dump joblib self.datamodule.x-scaler (.format "{}/{}-x.scale" model-path device-name))
      (.dump joblib self.datamodule.y-trafo  (.format "{}/{}-y.trafo" model-path device-name))
      (.dump joblib self.datamodule.y-scaler (.format "{}/{}-y.scale" model-path device-name))

      (when (get self.config "serialize")
        (let[path   self.model.cb-checkpoint.best-model-path
             model  (PreceptModule.load-from-checkpoint path)
             _      (.eval model)
             _      (.freeze model)
             trace (.to-torchscript model 
                                    :method "trace" 
                                    :example-inputs (torch.rand 1 num-x))]
        (trace.save (.format "{}/{}-model.pt" model-path device-name)))))))
