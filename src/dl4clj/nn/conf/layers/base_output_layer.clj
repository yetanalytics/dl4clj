(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/BaseOutputLayer.html"}
  dl4clj.nn.conf.layers.base-output-layer
  (:require [nd4clj.linalg.lossfunctions.loss-functions :as loss-functions]
            [dl4clj.nn.conf.layers.feed-forward-layer :as ff-layer])
  (:import [org.deeplearning4j.nn.conf.layers BaseOutputLayer BaseOutputLayer$Builder]))
;;deprecated
(defn builder [^BaseOutputLayer$Builder b {:keys [custom-loss-function ;; (string)
                                                  loss-function        ;; (lossFunction)
                                                  ]
                                           :or {}
                                           :as opts}]
  (let [b ^BaseOutputLayer$Builder (ff-layer/builder b opts)]
    (when custom-loss-function
      (.customLossFunction b custom-loss-function))
    (when loss-function
      (.lossFunction b (loss-functions/value-of loss-function)))
    b))
