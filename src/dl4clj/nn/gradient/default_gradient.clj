(ns ^{:doc "Default gradient implementation. Basically lookup table for ndarrays
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/gradient/DefaultGradient.html"}
    dl4clj.nn.gradient.default-gradient
  (:import [org.deeplearning4j.nn.gradient DefaultGradient]))

(defn constructor
  [& {:keys [flattened-gradient]
      :as opts}]
  (if (contains? opts :flattened-gradient)
    (DefaultGradient. flattened-gradient)
    (DefaultGradient.)))

(defn clear!
  "Clear residual parameters (useful for returning a gradient and then clearing old objects)"
  [this]
  (doto this (.clear)))

(defn flattening-order-for-variables
  "Return the gradient flattening order for the specified variable, or null if it is not explicitly set"
  [& {:keys [this variable]}]
  (.flatteningOrderForVariable this variable))

(defn get-gradient-for
  "The gradient for the given variable"
  [& {:keys [this variable]}]
  (.getGradientFor this variable))

(defn gradient
  "The full gradient as one flat vector"
  [& {:keys [this order]}]
  (if order
    (.gradient this order)
    (.gradient this)))

(defn gradient-for-variable
  "Gradient look up table"
  [this]
  (.gradientForVariable this))

(defn set-gradient-for!
  "Update gradient for the given variable; also (optionally) specify the order in which the array should be flattened to a row vector"
  [& {:keys [this variable new-gradient flattening-order]}]
  (if flattening-order
    (.setGradientFor this variable new-gradient flattening-order)
    (.setGradientFor this variable new-gradient)))
