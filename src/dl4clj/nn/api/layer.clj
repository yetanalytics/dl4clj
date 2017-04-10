(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.html"}
  dl4clj.nn.api.layer
  (:import [org.deeplearning4j.nn.api Layer])
  (:require [dl4clj.nn.conf.utils :refer [contains-many?]]
            [dl4clj.nn.api.model :refer :all]))

(defn activate
  "5 opts for triggering an activation
  1) supply training? (boolean), trigger with the last specified input
  2) supply input (INDArray), initialize the layer with the given input and return
   the activation for this layer given this input
  3) supply training-mode (keyword), Trigger an activation with the last specified input
  4) supply input and training?
  5) supply input and training-mode
   -both 4 and 5 initialize the layer with the given input and return the activation for this layer given this input"
  [& {:keys [this training? input training-mode]
      :as opts}]
  (cond-> this
    (contains-many? opts :this :input :training?) (.activate input training?)
    (contains-many? opts :this :input :training-mode) (.activate input training-mode)
    (and (contains-many? opts :this :input)
         (false? (contains-many? opts :training? :training-mode))) (.activate input)
    (and (contains-many? opts :this :training?)
         (false? (contains-many? opts :input :training-mode))) (.activate training?)
    (and (contains-many? opts :this :training-mode)
         (false? (contains-many? opts :input :training?))) (.activate training-mode)
    :else
    .activate))

(defn feed-forward-mask-array
  "Feed forward the input mask array, setting in in the layer as appropriate."
  [& {:keys [this mask-array mask-state batch-size]}]
  (.feedForwardMaskArray this mask-array mask-state batch-size))

(defn backprop-gradient
  "Calculate the gradient relative to the error in the next layer"
  [& {:keys [this epsilon]}]
  (.backpropGradient this epsilon))

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

(defn get-mask-array
  "get the mask array"
  [this]
  (.getMaskArray this))

(defn is-pretrain-layer?
  "Returns true if the layer can be trained in an unsupervised/pretrain manner (VAE, RBMs etc)"
  [this]
  (.isPretrainLayer this))

(defn pre-output
  "Raw activations"
  [& {:keys [this x training? training-mode]
      :as opts}]
  (cond-> this
    (contains-many? opts :this :training?) (.preOutput x training?)
    (contains-many? opts :this :training-mode) (.preOutput x training-mode)
    :else
    (.preOutput x)))

(defn set-index
  "Set the layer index."
  [& {:keys [this index]}]
  (.setIndex this (int index)))

(defn set-input
  "Get the layer input."
  [& {:keys [this input]}]
  (.setInput this input))

(defn set-input-mini-batch-size
  "Set current/last input mini-batch size.  Used for score and gradient calculations."
  [& {:keys [this size]}]
  (.setInputMiniBatchSize this (int size)))

(defn set-mask-array
  ""
  [& {:keys [this mask-array]}]
  (.setMaskArray this mask-array))

(defn transpose
  "Return a transposed copy of the weights/bias (this means reverse the number of inputs and outputs on the weights)"
  [^Layer this]
  (.transpose this))

(defn layer-type
  "Returns the layer type"
  [^Layer this]
  (.type this))
