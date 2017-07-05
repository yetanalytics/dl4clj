(ns dl4clj.nn.api.multi-layer-network
  (:import [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.api Layer])
  (:require [dl4clj.utils :refer [contains-many? array-of]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn activate-selected-layers
  "Calculate activation for few layers at once. Suitable for autoencoder partial activation

  returns the activation from the last layer

   :from (int), starting layer idx

   :to (int), ending layer idx

   :input (INDArray or vec), the input to propagate through the layers"
  [& {:keys [mln from to input]}]
  (.activateSelectedLayers mln from to (vec-or-matrix->indarray input)))

(defn activate-from-prev-layer
  "Calculate activation from previous layer including pre processing where necessary

  :current-layer-idx (int), the index of the current layer
   - you will get the activation from the layer directly before this one

  :input (INDArray or vec), the input to propagate through the layers

  :training? (boolean), is this training mode?"
  [& {:keys [mln current-layer-idx input training?]}]
  (.activationFromPrevLayer mln current-layer-idx
                            (vec-or-matrix->indarray input) training?))

(defn clear-layer-mask-arrays!
  "Remove the mask arrays from all layers.

  returns the multi layer network after the mutation"
  [mln]
  (doto mln
    (.clearLayerMaskArrays)))

(defn compute-z
  "if you only supply training?: Compute input linear transformation (z) of the output layer
  if you supply training? and input: Compute activations from input to output of the output layer
   - both ways return the list of activations for each layer

  :training? (boolean), training mode?

  :input (INDArray or vec), the input to propagate through the network for calcing activations"
  [& {:keys [mln training? input]
      :as opts}]
  (if (contains? opts :input)
    (.computeZ mln (vec-or-matrix->indarray input) training?)
    (.computeZ mln training?)))

(defn get-epsilon
  "returns epsilon for a given multi-layer-network (mln)"
  [mln]
  (.epsilon mln))

(defn feed-forward
  "if :features-mask and :labels-mask supplied:

   Compute the activations from the input to the output layer,
   given mask arrays (that may be null) The masking arrays are used in situations
   such an one-to-many and many-to-one rucerrent neural network (RNN) designs,
   as well as for supporting time series of varying lengths within the same minibatch for RNNs.

  else, just compute the activations from the input to the output layer

  :train? (boolean), is this training mode?

  :input (INDArray or vec), the input to be propagated through the network

  :features-mask (INDArray or vec), mask for the input features

  :labels-mask (INDArray or vec), mask for the labels"
  [& {:keys [mln train? input features-mask labels-mask]
      :as opts}]
  (let [i (vec-or-matrix->indarray input)]
   (cond (contains-many? opts :input :features-mask :labels-mask)
         (.feedForward mln i
                       (vec-or-matrix->indarray features-mask)
                       (vec-or-matrix->indarray labels-mask))
        (contains-many? opts :input :train?)
        (.feedForward mln i train?)
        (contains? opts :input)
        (.feedForward mln i)
        (contains? opts :train?)
        (.feedForward mln train?)
        :else
        (.feedForward mln))))

(defn feed-forward-to-layer
  "Compute the activations from the input to the specified layer.
   - if input is not supplied, uses the currently set input for the mln

  :layer-idx (int), the index of the layer you want the input propagated through

  :train? (boolean), are we in training mode?

  :input (INDArray or vec), the input to propagate through the specified layer

  Note: the returned output list contains the original input at idx 0"
  [& {:keys [mln layer-idx train? input]
      :as opts}]
  (let [i (vec-or-matrix->indarray input)]
    (cond (contains-many? opts :layer-idx :train? :input)
          (.feedForwardToLayer mln layer-idx i train?)
          (contains-many? opts :layer-idx :input)
          (.feedForwardToLayer mln layer-idx i)
          (contains-many? opts :layer-idx :train?)
          (.feedForwardToLayer mln layer-idx train?)
          :else
          (assert false "you must supply a mln, a layer-idx and either/both train? and input"))))

(defn get-default-config
  "gets the default config for the multi-layer-network"
  [mln]
  (.getDefaultConfiguration mln))

(defn get-input
  "return the input to the mln"
  [mln]
  (.getInput mln))

(defn get-layer
  "return the layer of the mln based on its position within the mln

  :layer-idx (int), the index of the layer you want to get from the mln

  :layer-name (str), the name of the layer you want to get from the mln"
  [& {:keys [mln layer-idx layer-name]
      :as opts}]
  (cond (contains? opts :layer-idx)
        (.getLayer mln layer-idx)
        (contains? opts :layer-name)
        (.getLayer mln layer-name)
        :else
        (assert false "you must supply a mln and either the layer's name or index")))

(defn get-layer-names
  "return a list of the layer names in the mln"
  [mln]
  (.getLayerNames mln))

(defn get-layers
  "returns an array of the layers within the mln"
  [mln]
  (.getLayers mln))

(defn get-layer-wise-config
  "returns the configuration for the layers in the mln"
  [mln]
  (.getLayerWiseConfigurations mln))

(defn get-mask
  "return the mask array used in this mln"
  [mln]
  (.getMask mln))

(defn get-n-layers
  "get the number of layers in the mln"
  [mln]
  (.getnLayers mln))

(defn get-output-layer
  "returns the output layer of the mln"
  [mln]
  (.getOutputLayer mln))

(defn get-updater
  "return the updater used in this mln"
  [mln]
  (.getUpdater mln))

(defn init-gradients-view!
  "initializes the flattened gradients array (used in backprop) and
  sets the appropriate subset in all layers.

  - this gets called behind the scene when using fit!"
  [mln]
  (doto mln
    (.initGradientsView)))

(defn get-mln-input
  "returns the input/feature matrix for the model"
  [mln]
  (.input mln))

(defn is-init-called?
  "was the model initialized"
  [mln]
  (.isInitCalled mln))

(defn print-config
  "Prints the configuration and returns the mln"
  [mln]
  (doto mln
    (.printConfiguration)))

(defn rnn-activate-using-stroed-state
  "returns the activation of the rnn given its most recent state
   - does not modify the RNN layer state, pure fn

  :input (INDArray or vec), the input fed to the rnn

  :training? (boolean), is this training mode?

  :store-last-for-tbptt? (boolean), set to true if used as part of truncated bptt training

  returns the activations for each layer
   - the input is idx 0, followed by the activations"
  [& {:keys [mln input training? store-last-for-tbptt?]
      :as opts}]
  (assert (contains-many? opts :input :training? :store-last-for-tbptt?)
          "you must supply a mln, the input to the model, if this is during training
or evaluation and if we want to store the previous state for truncated backprop")
  (.rnnActivateUsingStoredState mln (vec-or-matrix->indarray input) training? store-last-for-tbptt?))

(defn rnn-clear-prev-state!
  "clear the previous state of the rnn layers if any and return the mln"
  [mln]
  (doto mln
    (.rnnClearPreviousState)))

(defn rnn-get-prev-state
  "get the state of the rnn layer given its index in the mln

  :layer-idx (int), the index of the rnn within the mln"
  [& {:keys [mln layer-idx]}]
  (.rnnGetPreviousState mln layer-idx))

(defn rnn-set-prev-state!
  "Set the state of the RNN layer and return the updated mln

  :layer-idx (int), the index of the rnn within the mln

  :state (map), {str INDArray}, The state to set the specified layer to

  returns the mln"
  [& {:keys [mln layer-idx state]
      :as opts}]
  (assert (contains-many? opts :layer-idx :state)
          "you must supply a layer-index for the layer in question within the mln and
a map of the desired state")
  (doto mln
    (.rnnSetPreviousState layer-idx state)))

(defn set-mln-input!
  "Note that if input isn't nil and the neuralNets are nil,
  this is a way of initializing the neural network, returns the mln

  :input (INDArray or vec), the input to the mln"
  [& {:keys [mln input]}]
  (doto mln
    (.setInput (vec-or-matrix->indarray input))))

(defn set-labels-mln!
  "sets the labels given an array of labels,
  returns the mln.

  :labels (INDArray or vec), the labels to be set"
  [& {:keys [mln labels]}]
  (doto mln
    (.setLabels (vec-or-matrix->indarray labels))))

(defn set-layers!
  "sets the layers of the mln in the order in which they appear in the supplied coll.

  :layers (coll), a collection of layers to add to the mln

  returns the mln"
  [& {:keys [mln layers]}]
  (doto mln
    (.setLayers (array-of :data layers
                          :java-type Layer))))

(defn set-layer-wise-config!
  "sets the configuration for a mln given a multi-layer configuration.
  returns the mln

  :mln (multi layer network), the multi layer network

  :multi-layer-conf (multi layer conf), the configuration for the multi layer network

  NOTE: you should not need this fn.  You can set the multi-layer-conf when creating your mln
   - see: new-multi-layer-network at the top of this ns"
  [& {:keys [mln multi-layer-conf]}]
  (doto mln
    (.setLayerWiseConfigurations multi-layer-conf)))

(defn set-mask!
  "set the mask, returns the mln

  :mask (INDArray or vec), the mask to set for the mln"
  [& {:keys [mln mask]}]
  (doto mln
    (.setMask (vec-or-matrix->indarray mask))))

(defn set-parameters!
  "set the paramters for this model (mln).
   - This is used to manipulate the weights and biases across all neuralNets
     (including the output layer)

  :params (INDArray or vec), a parameter vector equal 1,numParameters

  returns the mln"
  [& {:keys [mln params]}]
  (doto mln
    (.setParameters (vec-or-matrix->indarray params))))

(defn set-score!
  "sets the score,

   :score (double), the score to set

  returns the mln"
  [& {:keys [mln score]}]
  (doto mln
    (.setScore score)))

(defn set-updater!
  "sets the updater for a given mln.

  :updater (ml-updater), the updater to use
   - see: dl4clj.nn.updater.multi-layer-updater

  returns the mln"
  [& {:keys [mln updater]}]
  (doto mln
    (.setUpdater updater)))

(defn update-mln!
  "Assigns the parameters of mln to the ones specified by another mln.
  This is used in loading from input streams, factory methods, etc
   - returns mln

  you can also use update! in the model interface ns"
  [& {:keys [mln other-mln]}]
  (doto mln (.update other-mln)))

(defn update-rnn-state-with-tbptt-state!
  "updates the rnn state to be that of the tbptt state.
  returns the mln."
  [mln]
  (doto mln
    (.updateRnnStateWithTBPTTState)))

(defn z-from-prev-layer
  "Compute input linear transformation (z) from previous layer
  Applies pre processing transformation where necessary

  :current-layer-idx (int), the current layer

  :input (INDArray or vec), the input

  :training? (boolean), are we in training mode?

  returns the activation from the previous layer"
  [& {:keys [mln current-layer-idx input training?]
      :as opts}]
  (assert (contains-many? opts :current-layer-idx :input :training?)
          "you must supply the index of the current layer, an input array and if this is for training or evaluation")
  (.zFromPrevLayer mln current-layer-idx (vec-or-matrix->indarray input) training?))
