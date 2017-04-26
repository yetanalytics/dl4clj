(ns ^{:doc "Center loss is similar to triplet loss except that it enforces intraclass
 consistency and doesn't require feed forward of multiple examples.
 Center loss typically converges faster for training ImageNet-based convolutional networks.
 ie. If example x is in class Y, ensure that embedding(x) is close to average(embedding(y)) for all examples y in Y

Implementation of the CenterLossOutputLayer class in dl4j.
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/training/CenterLossOutputLayer.html"}
    dl4clj.nn.layers.center-loss-output-layer
  (:import [org.deeplearning4j.nn.layers.training CenterLossOutputLayer])
  (:require [dl4clj.nn.api.classifier :refer :all]
            [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.layers.i-output-layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]
            [dl4clj.nn.layers.base-output-layer :refer :all]))

(defn new-center-loss-output-layer
  "creates a new center loss output layer by calling the CenterLossOutputLayer constructor.

  :conf is a neural network configuration
  :input is an INDArray of input values"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (CenterLossOutputLayer. conf input)
    (CenterLossOutputLayer. conf)))
