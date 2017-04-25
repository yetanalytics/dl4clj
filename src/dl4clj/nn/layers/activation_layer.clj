(ns ^{:doc "Activation Layer Used to apply activation on input and corresponding derivative on epsilon.
 Decouples activation from the layer type and ideal for cases when applying BatchNormLayer.
 For example, use identity activation on the layer prior to BatchNorm and apply this layer after the BatchNorm.
Implementation of the class ActivationLayer in dl4j.
all refered fns support activation-layer as the first arg
see https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/ActivationLayer.html"}
    dl4clj.nn.layers.activation-layer
  (:import [org.deeplearning4j.nn.layers ActivationLayer])
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.layers.base-layer :refer :all]))

(defn new-activation-layer
  "creates a new activation-layer by calling the ActivationLayer constructor.

  :conf is a neural network configuration
  :input is an INDArray of input values"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (ActivationLayer. conf input)
    (ActivationLayer. conf)))


;; see nn.api.layer, nn.api.model and nn.layers.base-layer
;; for a list of other interaction fns that activation-layer supports

;; an activation-layer can take the place of a base-layer in refered fns
