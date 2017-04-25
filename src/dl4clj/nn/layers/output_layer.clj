(ns ^{:doc "Output layer with different objective incooccurrences for different objectives.
This includes classification as well as prediction.
Implementation of the class OutputLayer in dl4j.
all refered fns support output-layer as the first arg
see https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/OutputLayer.html"}
    dl4clj.nn.layers.output-layer
  (:require [dl4clj.nn.api.classifier :refer :all]
            [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.layers.i-output-layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.layers.base-output-layer :refer [classify-input]]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]
            [dl4clj.nn.conf.utils :refer [contains-many?]])
  (:import [org.deeplearning4j.nn.layers OutputLayer]))

;; see the :refer :all namespaces for other interation fns that output-layer supports

(defn new-output-layer
  "creates a new output-layer by calling the OutputLayer constructor.

  :conf is a neural network configuration
  :input is an INDArray of input values"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (OutputLayer. conf input)
    (OutputLayer. conf)))
