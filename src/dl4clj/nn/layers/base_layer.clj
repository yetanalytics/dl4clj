(ns ^{:doc "A layer with a bias and activation function, implementation of the class BaseLayer in dl4j.
all refered fns support base-layer as the first arg
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/BaseLayer.html"}
    dl4clj.nn.layers.base-layer
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all])
  (:import [org.deeplearning4j.nn.layers BaseLayer]))

;; see nn.api.layer and nn.api.model for a list of other interaction fns that base-layer supports

(defn new-base-layer
  "creates a new base-layer by calling the BaseLayer constructor.

  :conf is a neural network configuration
  :input is an INDArray of input values"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (BaseLayer. conf input)
    (BaseLayer. conf)))

(defn calc-activation-mean
  "Calculate the mean representation for the activation for this layer"
  [base-layer]
  (.activationMean base-layer))

(defn calc-gradient
  "Calculate and return the gradient

  :layer-error is a gradient
  :activation is a INDArray of activation values for each neuron in the layer"
  [& {:keys [base-layer layer-error activation]}]
  (.calcGradient base-layer layer-error activation))

(defn derivative-activation
  "Take the derivative of the given input based on the activation

  input is an INDArray of input values to the layer"
  [& {:keys [base-layer input]}]
  (.derivativeActivation base-layer input))

(defn calc-error
  "Calculate error with respect to the current layer.

  :error-signal is an INDArray of error signals"
  [& {:keys [base-layer error-signal]}]
  (.error base-layer error-signal))

(defn get-input
  "returns the input to the layer"
  [base-layer]
  (.getInput base-layer))

(defn init-params!
  "Initialize the parameters"
  [base-layer]
  (doto base-layer
    (.initParams)))

(defn merge-layers!
  "Averages the given logistic regression from a mini batch (other layer)
  into the base-layer.

  :base-layer and :other-layer are both built layers from any-layer-builder
  :batch-size (int) size of the batch fed to the layer"
  [& {:keys [base-layer other-layer batch-size]}]
  (doto base-layer
    (.merge other-layer batch-size)))

(defn validate-input!
  "validates the input to the base-layer and returns the base-layer"
  [base-layer]
  (.validateInput base-layer))
