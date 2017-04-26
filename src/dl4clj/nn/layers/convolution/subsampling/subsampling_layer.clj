(ns ^{:doc "Subsampling layer. Used for downsampling a convolution
Implementation of the class SubsamplingLayer in dl4j.
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/convolution/subsampling/SubsamplingLayer.html"}
    dl4clj.nn.layers.convolution.subsampling.subsampling-layer
  (:import [org.deeplearning4j.nn.layers.convolution.subsampling SubsamplingLayer])
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

(defn new-subsampling-layer
  "creates a new convolution-1d-layer given a neural net conf and
  optionally some input data"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (SubsamplingLayer. conf input)
    (SubsamplingLayer. conf)))
