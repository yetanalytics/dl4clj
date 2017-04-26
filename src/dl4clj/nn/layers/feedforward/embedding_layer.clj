(ns ^{:doc "Embedding layer: feed-forward layer that expects single integers per example as
 input (class numbers, in range 0 to numClass-1) as input. This input has shape [numExamples,1]
 instead of [numExamples,numClasses] for the equivalent one-hot representation.
 Mathematically, EmbeddingLayer is equivalent to using a DenseLayer with a one-hot
 representation for the input; however, it can be much more efficient with a
large number of classes (as a dense layer + one-hot input does a matrix multiply with all but one value being zero).

NOTE: can only be used as the first layer for a network,
the weight rows can be considered a vector/embedding for each example

Implementation of the class EmbeddingLayer in dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/feedforward/embedding/EmbeddingLayer.html"}
    dl4clj.nn.layers.feedforward.embedding-layer
  (:import [org.deeplearning4j.nn.layers.feedforward.embedding EmbeddingLayer])
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

(defn new-embedding-layer
  "creates a new embedding-layer given a neural net configuration"
  [conf]
  (EmbeddingLayer. conf))
