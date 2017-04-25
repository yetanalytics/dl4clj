(ns ^{:doc "Implementation of the class DropoutLayer in dl4j.
all refered fns support dropout-layer as the first arg
see https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/DropoutLayer.html"}
    dl4clj.nn.layers.dropout-layer
  (:require [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]
            [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all])
  (:import [org.deeplearning4j.nn.layers DropoutLayer]))

(defn new-dropout-layer
  "creates a new dropout-layer by calling the DropoutLayer constructor.

  :conf is a neural network configuration
  :input is an INDArray of input values"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (DropoutLayer. conf input)
    (DropoutLayer. conf)))
