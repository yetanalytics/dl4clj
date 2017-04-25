(ns ^{:doc "Baseline class for any Neural Network used as a layer in a deep network.
Implementation of the class BasePretrainNetwork in dl4j.
all refered fns support base-pretrain-network as the first arg.
see https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/BasePretrainNetwork.html"}
    dl4clj.nn.layers.base-pretrain-network
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]])
  (:import [org.deeplearning4j.nn.layers BasePretrainNetwork]))

(defn new-base-pretrain-network
  "creates a new base-pretrain-network by calling the BasePretrainNetwork constructor.

  :conf is a neural network configuration
  :input is an INDArray of input values"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (BasePretrainNetwork. conf input)
    (BasePretrainNetwork. conf)))

(defn get-corrupted-input
  "Corrupts the given input by doing a binomial sampling given the corruption level

  :features is an INDArray of input features to the network
  :corruption-level (double) amount of corruption to apply."
  [& {:keys [base-pretrain-network features corruption-level]}]
  (.getCorruptedInput base-pretrain-network features corruption-level))

(defn sample-hidden-given-visible
  "Sample the hidden distribution given the visible

  :visible is an INDArray of the visibile distribution"
  [& {:keys [base-pretrain-network visible]}]
  (.sampleHiddenGivenVisible base-pretrain-network visible))

(defn sample-visible-given-hidden
  "Sample the visible distribution given the hidden

  :hidden is an INDArray of the hidden distribution"
  [& {:keys [base-pretrain-network hidden]}]
  (.sampleVisibleGivenHidden base-pretrain-network hidden))
