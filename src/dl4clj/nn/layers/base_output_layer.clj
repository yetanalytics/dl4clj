(ns ^{:doc "Output layer with different objective in co-occurrences for different objectives.
 This includes classification as well as prediction.  Implementation of the class BaseOutputLayer in dl4j.
A base-output-layer can be used when a fn requires either a layer or a base-layer as an arg.
A base-output-layer can also be used when a fn requires a model as an arg
see https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/BaseOutputLayer.html"}
    dl4clj.nn.layers.base-output-layer
  (:require [dl4clj.nn.api.classifier :refer :all]
            [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.layers.i-output-layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.conf.utils :refer [contains-many?]])
  (:import [org.deeplearning4j.nn.layers BaseOutputLayer]))

;; see the :refer :all namespaces for other interation fns that base-output-layer supports

(defn new-base-output-layer
  "creates a new base-output-layer by calling the BaseOutputLayer constructor.

  :conf is a neural network configuration
  :input is an INDArray of input values"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (BaseOutputLayer. conf input)
    (BaseOutputLayer. conf)))

(defn classify-input
  "classifies the input supplied"
  [& {:keys [base-output-layer training? input]
      :as opts}]
  (assert (contains? opts :base-output-layer) "you must supply an output-layer")
  (cond (contains-many? opts :training? :input)
        (.output base-output-layer input training?)
        (contains? opts :training?)
        (.output base-output-layer training?)
        (contains? opts :input)
        (.output base-output-layer input)
        :else
        (assert false "you must supply training? and/or input")))
