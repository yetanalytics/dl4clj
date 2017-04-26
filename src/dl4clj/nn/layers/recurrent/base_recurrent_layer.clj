(ns ^{:doc "Implementation of the class BaseRecurrentLayer in dl4j.
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/recurrent/BaseRecurrentLayer.html"}
    dl4clj.nn.layers.recurrent.base-recurrent-layer
  (:import [org.deeplearning4j.nn.layers.recurrent BaseRecurrentLayer])
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.api.layers.recurrent-layer :refer :all]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]))

(defn new-base-recurrent-layer
  "creates a new base recurrent layer given a neural net conf and
  optionally some input data"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (BaseRecurrentLayer. conf input)
    (BaseRecurrentLayer. conf)))
