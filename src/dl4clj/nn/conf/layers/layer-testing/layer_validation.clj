(ns dl4clj.nn.conf.layers.layer-testing.layer-validation
  (:import [org.deeplearning4j.nn.conf.layers LayerValidation]))

(defn constructor []
  (LayerValidation.))

(defn general-validation
  [& {:keys [layer-name layer use-regularization? use-drop-connect? drop-out
             l2 l2-bias l1 l1-bias dist]}]
  (.generalValidation layer-name layer use-regularization? use-drop-connect?
                      drop-out l2 l2-bias l1 l1-bias dist))

(defn updater-validation
  [& {:keys [layer-name layer momentum momentum-schedule adam-mean-decay
             adam-var-decay rho rms-decay epsilon]}]
  (.updaterValidation layer-name layer momentum momentum-schedule
                      adam-mean-decay adam-var-decay rho rms-decay
                      epsilon))
