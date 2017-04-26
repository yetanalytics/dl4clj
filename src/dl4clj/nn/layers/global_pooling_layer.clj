(ns ^{:doc "used to do pooling over time for RNNs, and 2d pooling for CNNs.
Supports the following PoolingTypes: SUM, AVG, MAX, PNORM
Global pooling layer can also handle mask arrays when dealing with variable length inputs.
 Mask arrays are assumed to be 2d, and are fed forward through the network during training or post-training forward pass:
- Time series: mask arrays are shape [minibatchSize, maxTimeSeriesLength] and contain values 0 or 1 only
- CNNs: mask have shape [minibatchSize, height] or [minibatchSize, width]. Important: the current implementation assumes that for CNNs + variable length (masking), the input shape is [minibatchSize, depth, height, 1] or [minibatchSize, depth, 1, width] respectively. This is the case with global pooling in architectures like CNN for sentence classification.
Behaviour with default settings:
- 3d (time series) input with shape [minibatchSize, vectorSize, timeSeriesLength] -> 2d output [minibatchSize, vectorSize]
- 4d (CNN) input with shape [minibatchSize, depth, height, width] -> 2d output [minibatchSize, depth]
Alternatively, by setting collapseDimensions = false in the configuration, it is possible to retain the reduced dimensions as 1s: this gives [minibatchSize, vectorSize, 1] for RNN output, and [minibatchSize, depth, 1, 1] for CNN output.

Implementation of the class GlobalPoolingLayer in dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/pooling/GlobalPoolingLayer.html"}
    dl4clj.nn.layers.global-pooling-layer
  (:import [org.deeplearning4j.nn.layers.pooling GlobalPoolingLayer])
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

(defn new-global-pooling-layer
  "creates a new global pooling layer given a neural net configuration"
  [conf]
  (GlobalPoolingLayer. conf))
