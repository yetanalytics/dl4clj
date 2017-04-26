(ns ^{:doc "Zero padding layer for convolutional neural networks. Allows padding to be done separately for top/bottom/left/right
Implementation of the class ZeroPaddingLayer in dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/convolution/ZeroPaddingLayer.html"}
    dl4clj.nn.layers.convolution.zero-padding-layer
  (:import [org.deeplearning4j.nn.layers.convolution ZeroPaddingLayer])
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

(defn new-zero-padding-layer
  "creates a new zero padding layer given a neural net conf"
  [conf]
  (ZeroPaddingLayer. conf))
