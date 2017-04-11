(ns dl4clj.nn.gradient.default-gradient
  (:import [org.deeplearning4j.nn.gradient DefaultGradient]))

(defn constructor
  ([]
   (DefaultGradient.))
  ([flattened-gradient]
   (DefaultGradient. flattened-gradient)))

(defn clear
  "Clear residual parameters (useful for returning a gradient and then clearing old objects)"
  [this]
  (.clear this))

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

(defn set-gradient-for
  "Update gradient for the given variable; also (optionally) specify the order in which the array should be flattened to a row vector"
  [& {:keys [this variable new-gradient flattening-order]}]
  (if flattening-order
    (.setGradientFor this variable new-gradient flattening-order)
    (.setGradientFor this variable new-gradient)))

(defn to-string
  [this]
  (.toString this))
