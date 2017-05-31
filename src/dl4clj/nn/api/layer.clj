(ns ^{:doc "fns from the dl4j Interface for a layer of a neural network.
 This has an activation function, an input and output size, weights, and a bias
 see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.html"}
  dl4clj.nn.api.layer
  (:import [org.deeplearning4j.nn.api Layer])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.nn.conf.constants :as enum]))

(defn activate
  "6 opts for triggering an activation
  1) only supply the layer, Trigger an activation with the last specified input
  2) supply training? (boolean), trigger with the last specified input
  3) supply input (INDArray), initialize the layer with the given input and return
   the activation for this layer given this input
  4) supply training-mode (keyword), Trigger an activation with the last specified input
    - keyword is one of :test or :train
  5) supply input and training?
  6) supply input and training-mode
   -both 4 and 5 initialize the layer with the given input and return the activation for this layer given this input"
  [& {:keys [layer training? input training-mode]
      :as opts}]
  (assert (contains? opts :layer) "you must supply a layer to activate")
  (cond (contains-many? opts :input :training?)
        (.activate layer input training?)
        (contains-many? opts :input :training-mode)
        (.activate layer input (enum/value-of {:layer-training-mode training-mode}))
        (contains? opts :input)
        (.activate layer input)
        (contains? opts :training?)
        (.activate layer training?)
        (contains? opts :training-mode)
        (.activate layer (enum/value-of {:layer-training-mode training-mode}))
        :else
        (.activate layer)))

(defn feed-forward-mask-array
  "Feed forward the input mask array, setting in in the layer as appropriate.
   mask-array (INDArray of mask values),
   mask-state (keyword), either :active or :passthrough
    - :active = apply mask to activations and errors.
    - :passthrough = feed forward the input mask (if/when necessary) but don't actually apply it.
    - Note: Masks should not be applied in all cases, depends on the network configuration
  batch-size (integer) the minibatch size to use"
  [& {:keys [layer mask-array mask-state batch-size]}]
  (.feedForwardMaskArray layer mask-array
   (enum/value-of {:mask-state mask-state})
   batch-size))

(defn backprop-gradient
  "Calculate the gradient relative to the error in the next layer
   epsilon is an INDArray"
  [& {:keys [layer epsilon]}]
  (.backpropGradient layer epsilon))

(defn calc-l1
  "Calculate the l1 regularization term. 0.0 if regularization is not used."
  [& {:keys [layer backprop-only-params?]}]
  (.calcL1 layer backprop-only-params?))

(defn calc-l2
  "Calculate the l2 regularization term. 0.0 if regularization is not used."
  [& {:keys [layer backprop-only-params?]}]
  (.calcL2 layer backprop-only-params?))

(defn clone
  "Clone the layer"
  [layer]
  (.clone layer))

(defn get-index
  "Get the layer index."
  [layer]
  (.getIndex layer))

(defn get-input-mini-batch-size
  "Get current/last input mini-batch size, as set by set-input-mini-batch-size!"
  [layer]
  (.getInputMiniBatchSize layer))

(defn get-listeners
  "Get the iteration listeners for this layer."
  [layer]
  (.getListeners layer))

(defn get-mask-array
  "get the mask array"
  [layer]
  (.getMaskArray layer))

(defn is-pretrain-layer?
  "Returns true if the layer can be trained in an unsupervised/pretrain manner (VAE, RBMs etc)"
  [layer]
  (.isPretrainLayer layer))

(defn pre-output
  "returns an array of Raw activations given the features and layer

  -required args:

  layer: the built layer from any-layer-builder
  features: an INDArray of input features (the x values)

  -optional args:

  training? (boolean) are we training a layer or testing it?
  training-mode (keyword), one of :test or :train"
  [& {:keys [layer features training? training-mode]
      :as opts}]
  (assert (contains-many? opts :layer :features) "you must supply a layer and an INDArray of input features")
  (cond (contains? opts :training?)
        (.preOutput layer features training?)
        (contains? opts :training-mode)
        (.preOutput layer features (enum/value-of {:layer-training-mode training-mode}))
        :else
        (.preOutput layer features)))

(defn set-index!
  "Set the layer index and returns the layer.
  :index should be an integer"
  [& {:keys [layer index]}]
  (doto layer
    (.setIndex (int index))))

(defn set-input!
  "set the layer's input and return the layer"
  [& {:keys [layer input]}]
  (doto layer
    (.setInput input)))

(defn set-input-mini-batch-size!
  "Set current/last input mini-batch size and return the layer.
  Used for score and gradient calculations."
  [& {:keys [layer size]}]
  (doto layer
    (.setInputMiniBatchSize size)))

(defn set-layer-listeners!
  "Set the iteration listeners for the supplied layer and returns the layer.
  listeners is a collection or array of iteration listeners"
  [& {:keys [layer listeners]}]
  (doto layer
    (.setListeners listeners)))

(defn set-mask-array!
  "Set the mask array for this layer and return the layer
  mask-array is an INDArray"
  [& {:keys [layer mask-array]}]
  (doto layer
    (.setMaskArray mask-array)))

(defn transpose
  "Return a transposed copy of the weights/bias
  (this means reverse the number of inputs and outputs on the weights)"
  [layer]
  (.transpose layer))

(defn layer-type
  "Returns the layer type"
  [layer]
  (.type layer))
