(ns ^{:doc "LossLayer is a flexible output layer that performs a loss function on an input without MLP logic.
Implementation of the class LossLayer in dl4j.
all refered fns support loss-layer as the first arg
see https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/LossLayer.html"}
    dl4clj.nn.layers.loss-layer
  (:require [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]
            [dl4clj.nn.api.classifier :refer :all]
            [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.layers.i-output-layer :refer :all]
            [dl4clj.nn.api.model :refer :all])
  (:import [org.deeplearning4j.nn.layers LossLayer]))

(defn new-loss-layer
  "creates a new loss-layer by calling the LossLayer constructor.

  :conf is a neural network configuration
  :input is an INDArray of input values"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (LossLayer. conf input)
    (LossLayer. conf)))
