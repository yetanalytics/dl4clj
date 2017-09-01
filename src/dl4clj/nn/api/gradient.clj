(ns dl4clj.nn.api.gradient
  (:import [org.deeplearning4j.nn.gradient DefaultGradient])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn clear!
  "Clear residual parameters (useful for returning a gradient and then clearing old objects)"
  [grad]
  (doto grad (.clear)))

(defn flattening-order-for-variables
  "Return the gradient flattening order for the specified variable, or null if it is not explicitly set"
  [& {:keys [grad variable]}]
  (.flatteningOrderForVariable grad variable))

(defn get-gradient-for
  "The gradient for the given variable"
  [& {:keys [grad variable]}]
  (.getGradientFor grad variable))

(defn gradient
  "The full gradient as one flat vector"
  [& {:keys [grad order]}]
  (if order
    (.gradient grad order)
    (.gradient grad)))

(defn gradient-for-variable
  "Gradient look up table"
  [grad]
  (.gradientForVariable grad))

(defn set-gradient-for!
  "Update gradient for the given variable; also (optionally) specify the order in which the array should be flattened to a row vector"
  [& {:keys [grad variable new-gradient flattening-order]}]
  (let [ng (vec-or-matrix->indarray new-gradient)]
    (if flattening-order
      (doto grad (.setGradientFor variable ng flattening-order))
      (doto grad (.setGradientFor variable ng)))))
