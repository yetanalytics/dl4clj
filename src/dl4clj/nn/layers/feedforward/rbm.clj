(ns ^{:doc "Restricted Boltzmann Machine. Markov chain with gibbs sampling.
 Supports the following visible units: binary gaussian softmax linear
 Supports the following hidden units: rectified binary gaussian softmax linear
 Based on Hinton et al.'s work
 Implementation of the RBM class in dl4j
 see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/feedforward/rbm/RBM.html"}
    dl4clj.nn.layers.feedforward.rbm
  (:import [org.deeplearning4j.nn.layers.feedforward.rbm RBM])
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.layers.base-pretrain-network :refer [get-corrupted-input
                                                            sample-hidden-given-visible
                                                            sample-visible-given-hidden]]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]
            [dl4clj.nn.conf.utils :refer [contains-many?]]))

(defn new-rbm
  "creates a new rbm layer given a neural net config and optinally
  some input data."
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (RBM. conf input)
    (RBM. conf)))

(defn gibbs-sampling-step
  "Gibbs sampling step: hidden ---> visible ---> hidden
  returns the expected values and samples of both the visible samples given
  the hidden and the new hidden input and expected values"
  [& {:keys [rbm hidden-input]}]
  (.gibbhVh rbm hidden-input))

(defn prop-down
  "Calculates the activation of the hidden: (activation (h * W + vbias))"
  [& {:keys [rbm hidden-input]}]
  (.propDown rbm hidden-input))

(defn prop-up
  "Calculates the activation of the visible : sigmoid(v * W + hbias)"
  [& {:keys [rbm visible-input training?]
      :as opts}]
  (assert (contains-many? opts :rbm :visible-input)
          "you must supply a rbm and the visible input")
  (if (contains? opts :training?)
    (.propUp rbm visible-input training?)
    (.propUp rbm visible-input)))

(defn prop-up-derivative
  "derivative of the prop-up activation"
  [& {:keys [rbm prop-up-vals]}]
  (.propUpDerivative rbm prop-up-vals))
