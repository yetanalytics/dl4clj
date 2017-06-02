(ns ^{:doc "1D (temporal) subsampling layer. his layer accepts RNN (not CNN) InputTypes,
This approach treats a multivariate time series with L timesteps and P variables
 as an L x 1 x P image (L rows high, 1 column wide, P channels deep). The kernel should be H
Implementation of the class Subsampling1DLayer in dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/convolution/subsampling/Subsampling1DLayer.html"}
    dl4clj.nn.layers.convolution.subsampling.subsampling-1d-layer
  (:import [org.deeplearning4j.nn.layers.convolution.subsampling Subsampling1DLayer])
  #_(:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]))

(defn new-subsampling-1d-layer
  "creates a new subsampling-1d-layer given a neural net conf and
  optionally some input data"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (Subsampling1DLayer. conf input)
    (Subsampling1DLayer. conf)))
