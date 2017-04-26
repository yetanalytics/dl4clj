(ns ^{:doc "Implementation of the class DenseLayer in dl4j.
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/feedforward/dense/DenseLayer.html"}
    dl4clj.nn.layers.feedforward.dense-layer
  (:import [org.deeplearning4j.nn.layers.feedforward.dense DenseLayer])
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]))

(defn new-dense-layer
  "creates a new dense layer given a neural net conf and
  optionally some input data"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (DenseLayer. conf input)
    (DenseLayer. conf)))
