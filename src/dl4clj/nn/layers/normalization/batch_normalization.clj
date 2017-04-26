(ns ^{:doc "Batch normalization layer.  Implementation of the class BatchNormalization in dl4j.
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/normalization/BatchNormalization.html"}
    dl4clj.nn.layers.normalization.batch-normalization
  (:import [org.deeplearning4j.nn.layers.normalization BatchNormalization])
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

(defn new-batch-normalization-layer
  "create a new batch normalization layer given a neural network configuration"
  [conf]
  (BatchNormalization. conf))

(defn get-shape
  "returns an integer array of the shape after passing through the normalization layer"
  [& {:keys [batch-norm features]}]
  (.getShape batch-norm features))
