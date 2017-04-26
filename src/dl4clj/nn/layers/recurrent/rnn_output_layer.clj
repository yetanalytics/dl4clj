(ns ^{:doc "Recurrent Neural Network Output Layer.
Handles calculation of gradients etc for various objective functions.
Functionally the same as OutputLayer, but handles output and label reshaping automatically.
Input and output activations are same as other RNN layers:
 3 dimensions with shape [miniBatchSize,nIn,timeSeriesLength]
 and [miniBatchSize,nOut,timeSeriesLength] respectively.

Implementation of the class RnnOutputLayer in dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/recurrent/RnnOutputLayer.html"}
    dl4clj.nn.layers.recurrent.rnn-output-layer
  (:import [org.deeplearning4j.nn.layers.recurrent RnnOutputLayer])
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.api.classifier :refer :all]
            [dl4clj.nn.api.layers.i-output-layer :refer :all]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]
            [dl4clj.nn.layers.base-output-layer :refer [classify-input]]))

(defn new-base-rnn-output-layer
  "creates a new base rnn output layer given a neural net conf and
  optionally some input data"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (RnnOutputLayer. conf input)
    (RnnOutputLayer. conf)))
