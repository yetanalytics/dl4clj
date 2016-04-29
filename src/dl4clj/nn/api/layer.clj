(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.html"}
  dl4clj.nn.api.layer
  (:refer-clojure :exclude [type])
  (:import [org.deeplearning4j.nn.api Layer]
           [org.nd4j.linalg.api.ndarray INDArray]))

(defmulti activate (fn [this & more] (mapv clojure.core/type more)))
(defmethod activate []
  [^Layer this]
  (.activate this))
(defmethod activate [java.lang.Boolean]
  [^Layer this ^java.lang.Boolean training]
  (.activate this training))
(defmethod activate [INDArray]
  [^Layer this ^INDArray input]
  (.activate this input))
(defmethod activate [INDArray java.lang.Boolean]
  [^Layer this ^INDArray input ^Boolean training]
  (.activate this input training))
(defmethod activate [INDArray java.lang.Boolean]
  [^Layer this ^INDArray input training]
  (.activate this input))
;; (defmethod activate [Layer.TrainingMode]
;;   [^Layer this ^Layer.TrainingMode training]
;;   (.activate this training))

(defn activation-mean
  "Calculate the mean representation for the activation for this layer"
  [^Layer this]
  (.activationMean this))

(defn backprop-gradient
  "Calculate the gradient relative to the error in the next layer"
  [^Layer this ^INDArray epsilon]
  (.backpropGradient this epsilon))

(defn calc-gradient
  "Calculate the gradient"
  [^Layer this layer-error INDArray ind-array]
  (.calcGradient this layer-error ind-array))

(defn calc-l1
  "Calculate the l1 regularization term. 0.0 if regularization is not used."
  [^Layer this]
  (.calcL1 this))

(defn calc-l2
  "Calculate the l2 regularization term. 0.0 if regularization is not used."
  [^Layer this]
  (.calcL2 this))

(defn clone
  "Clone the layer"
  [^Layer this]
  (.clone this))

(defn derivative-activation
  "Take the derivative of the given input based on the activation"
  [^Layer this ^INDArray input]
  (.derivativeActivation this input)) 

(defn error
  "Calculate error with respect to the current layer."
  [^Layer this ^INDArray input]
  (.error this input))

(defn get-index
  "Get the layer index."
  [^Layer this]
  (.getIndex this))

(defn get-input-mini-batch-size
  "Get current/last input mini-batch size, as set by setInputMiniBatchSize(int)"
  [^Layer this]
  (.setInputMiniBatchSize this))

(defn get-listeners
  "Get the iteration listeners for this layer."
  [^Layer this]
  (.getListeners this))

(defn merge
  "Parameter averaging"
  [^Layer this ^Layer layer batch-size]
  (.merge this layer batch-size))

(defn pre-output
  "Raw activations"
  [^Layer this ^INDArray x]
  (.preOutput this x))

(defn pre-output
  "Raw activations"
  [^Layer this ^INDArray x training]
  (.preOutput this x (boolean training)))

;; (defn preOutput
;;   "Raw activations"
;;   [^Layer this ^INDArray x, ^Layer.TrainingMode training]
;;   (.preOutput this x training))

(defn set-index
  "Set the layer index."
  [^Layer this index]
  (.setIndex this (int index)))

(defn set-input
  "Get the layer input."
  [^Layer this ^INDArray input]
  (.setInput this input))

(defn set-input-mini-batch-size
  "Set current/last input mini-batch size.  Used for score and gradient calculations."
  [^Layer this size]
  (.setInputMiniBatchSize this (int size)))

(defn set-listeners
  "Set the iteration listeners for this layer."
  [^Layer this listeners]
  (.setListeners this listeners))

(defn set-mask-array
  ""
  [^Layer this ^INDArray mask-array]
  (.setMaskArray this mask-array))

(defn transpose
  "Return a transposed copy of the weights/bias (this means reverse the number of inputs and outputs on the weights)"
  [^Layer this]
  (.transpose this))

(defn type
  "Returns the layer type"
  [^Layer this]
  (.type this))

(defn update
  "Update layer weights and biases with gradient change"
  ([^Layer this gradient]
   (.update this gradient))
  ([^Layer this ^INDArray gradient param-type]
   (.update this gradient param-type)))
 
