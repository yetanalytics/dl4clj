(ns dl4clj.nn.multilayer.multi-layer-network
  (:require [dl4clj.nn.conf.utils :refer [contains-many?]])
  (:import [org.deeplearning4j.nn.multilayer MultiLayerNetwork]))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/multilayer/MultiLayerNetwork.html
;; add bangs for fns that modify the mln
;; write tests for these methods

(defn multi-layer-network
  "constructor for a multi-layer-network given a config and optionaly
  some params (INDArray)"
  ([conf]
   (MultiLayerNetwork. conf))
  ([conf params]
   (MultiLayerNetwork. conf params)))

(defn accumulate-score
  "Sets a rolling tally for the score."
  [mln accum]
  (doto mln
    (.accumulateScore accum)))

(defn activate
  "Triggers the activation of the last hidden layer ie: not logistic regression"
  ;; add docs for the various opts
  [mln & {:keys [training? input
                 layer-idx training-mode]
          :as opts}]
  (cond (contains-many? opts :input :training-mode)
        (.activate mln input training-mode)
        (contains-many? opts :input :training?)
        (.activate mln input training?)
        (contains-many? opts :input :layer-idx)
        (.activate mln layer-idx input)
        (contains? opts :training?)
        (.activate mln training?)
        (contains? opts :input)
        (.activate mln input)
        (contains? opts :layer-idx)
        (.activate mln layer-idx)
        (contains? opts :training-mode)
        (.activate mln training-mode)
        :else
        (.activate mln)))

(defn activate-selected-layers
  "Calculate activation for few layers at once."
  [mln & {:keys [from to input]}]
  (.activateSelectedLayers mln from to input))

(defn activate-from-prev-layer
  "Calculate activation from previous layer including pre processing where necessary"
  [mln & {:keys [current-layer-idx input training?]}]
  (.activationFromPrevLayer mln current-layer-idx input training?))

(defn activation-mean
  "Calculate the mean representation for the activation for this layer"
  [mln]
  (.activationMean mln))

(defn apply-learning-rate-score-decay
  "Update learningRate using for this model."
  [mln]
  (doto mln
    (.applyLearningRateScoreDecay)))

(defn back-prop-gradient
  "Calculate the gradient relative to the error in the next layer"
  [mln epsilon]
  (.backpropGradient mln epsilon))

(defn get-batch-size
  "The current inputs batch size"
  [mln]
  (.batchSize mln))

(defn calc-gradient
  "Calculate the gradient"
  [mln & {:keys [layer-error activation]}]
  (.calcGradient mln layer-error activation))

(defn calc-l1
  "Calculate the l1 regularization term
  0.0 if regularization is not used."
  [mln backprop-params-only?]
  (.calcL1 mln backprop-params-only?))

(defn calc-l2
  "Calculate the l2 regularization term
  0.0 if regularization is not used."
  [mln backprop-params-only?]
  (.calcL2 mln backprop-params-only?))

(defn clear
  "Clear the inputs."
  [mln]
  (doto mln
    (.clear)))

(defn clear-layer-mask-arrays
  "Remove the mask arrays from all layers."
  [mln]
  (doto mln
    (.clearLayerMaskArrays)))

(defn clone
  "Clone the layer" ;; is it cloning the layer or model? test this
  [mln]
  (.clone mln))

(defn compute-gradient-and-score
  "Update the score"
  [mln]
  (doto mln
    (.computeGradientAndScore mln)))

(defn compute-z
  "if you only supply training?: Compute input linear transformation (z) of the output layer
  if you supply training? and input: Compute activations from input to output of the output layer"
  [mln & {:keys [training? input]
          :as opts}]
  (if (contains? opts :input)
    (.computeZ mln input training?)
    (.computeZ mln training?)))

(defn get-configuration
  "The configuration for the neural network"
  [mln]
  (.conf mln))

(defn derivative-activation
  "Take the derivative of the given input based on the activation"
  [mln input]
  (.derivativeActivation mln input))

(defn do-evaluation
  "Perform evaluation using an arbitrary IEvaluation instance."
  [mln & {:keys [iterator evaluation]}]
  (.doEvaluation mln iterator evaluation))

(defn get-epsilon
  "returns epsilon for a given multi-layer-network (mln)"
  [mln]
  (.epsilon mln))

(defn calc-error
  "Calculate error with respect to the current layer."
  [mln error-signal]
  (.error mln error-signal))

(defn evaluate-classification
  "if you only supply mln and iterator: Evaluate the network (classification performance)
  if you supply mln, iterator and labels-list: Evaluate the network on the provided data set.
  if you supply all args: Evaluate the network (for classification) on the provided data set,
                          with top N accuracy in addition to standard accuracy."
  [mln iterator & {:keys [labels-list top-n]
                   :as opts}]
  (cond (contains-many? opts :labels-list :top-n)
        (.evaluate mln iterator labels-list top-n)
        (contains? opts :labels-list)
        (.evaluate mln iterator labels-list)
        :else
        (.evaluate mln iterator)))

(defn evaluate-regression
  "Evaluate the network for regression performance"
  [mln iterator]
  (.evaluateRegression mln iterator))

(defn evaluate-roc
  "Evaluate the network (must be a binary classifier) on the specified data, using the ROC class"
  [mln & {:keys [iterator roc-threshold-steps]}]
  (.evaluateROC mln iterator roc-threshold-steps))

(defn evaluate-roc-multi-class
  "Evaluate the network on the specified data, using the ROCMultiClass class"
  [mln {:keys [iterator roc-threshold-steps]}]
  (.evaluateROCMultiClass mln iterator roc-threshold-steps))

(defn f1-score
  "Sets the input and labels and returns a score for the prediction wrt true labels"
  [mln & {:keys [ds input labels]
          :as opts}]
  (if (contains? opts :ds)
    (.f1Score mln ds)
    (.f1Score mln input labels)))

(defn feed-forward
  "if :features-mask and :labels-mask supplied:

   Compute the activations from the input to the output layer,
   given mask arrays (that may be null) The masking arrays are used in situations
   such an one-to-many and many-to-one rucerrent neural network (RNN) designs,
   as well as for supporting time series of varying lengths within the same minibatch for RNNs.

  else, just compute the activations from the input to the output layer"
  [mln & {:keys [train? input features-mask labels-mask]
          :as opts}]
  (cond (contains-many? opts :input :features-mask :labels-mask)
        (.feedForward mln input features-mask labels-mask)
        (contains-many? opts :input :train?)
        (.feedForward mln input train?)
        (contains? opts :input)
        (.feedForward mln input)
        (contains? opts :train?)
        (.feedForward mln train?)
        :else
        (.feedForward mln)))

(defn feed-forward-mask-array
  "Feed forward the input mask array, setting in in the layer as appropriate."
  [mln & {:keys [mask-array current-mask-state mini-batch-size]}]
  (.feedForwardMaskArray mln mask-array current-mask-state mini-batch-size))

(defn feed-forward-to-layer
  "Compute the activations from the input to the specified layer.

  Note: the returned output list contains the original input"
  [mln & {:keys [layer-idx train? input]
          :as opts}]
  (cond (contains-many? opts :layer-idx :train? :input)
        (.feedForwardToLayer mln layer-idx input train?)
        (contains-many? opts :layer-idx :input)
        (.feedForwardToLayer mln layer-idx input)
        (contains-many? opts :layer-idx :train?)
        (.feedForwardToLayer mln layer-idx train?)
        :else
        (assert false "you must supply a mln, a layer-idx and either/both train? and input")))

(defn fine-tune
  "Run SGD based on the given labels"
  [mln]
  (doto mln
    (.finetune)))

(defn fit
  "Fit/train the model"
  [mln & {:keys [ds iterator data labels
                 features features-mask labels-mask
                 examples label-idxs]
          :as opts}]
  (cond (contains-many? opts :features :labels :features-mask
                        :labels-mask)
        (.fit mln features labels features-mask labels-mask)
        (contains-many? opts :examples :label-idxs)
        (.fit mln examples label-idxs)
        (contains-many? opts :data :labels)
        (.fit mln data labels)
        (contains? opts :data)
        (.fit mln data)
        (contains? opts :iterator)
        (.fit mln iterator)
        (contains? opts :ds)
        (.fit mln ds)
        :else
        (.fit mln))
  mln)

(defn get-default-config
  "gets the default config for the multi-layer-network"
  [mln]
  (.getDefaultConfiguration mln))

(defn get-idx
  "Get the layer index."
  [mln]
  (.getIndex mln))

(defn get-input
  "return the input to the mln"
  [mln]
  (.getInput mln))

(defn get-input-mini-batch-size
  "return the input mini batch size"
  [mln]
  (.getInputMiniBatchSize mln))

(defn get-labels
  "return an array of labels"
  [mln]
  (.getLabels mln))

(defn get-layer-by-idx
  "return the layer of the mln based on its position within the mln"
  [mln & {:keys [idx layer-name]
        :as opts}]
  (cond (and (contains? opts :idx)
             (integer? idx))
        (.getLayer mln idx)
        (and (contains? opts :layer-name)
             (string? layer-name))
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

(defn get-listeners
  "Get the iteration listeners for this layer."
  [mln]
  (.getListeners mln))

(defn get-mask
  "return the mask array used in this mln"
  ;; look into what exactly this does, no doc string in docs
  [mln]
  (.getMask mln))

(defn get-mask-array
  "return the mask array used in this mln"
  ;; look into what exactly this does, no doc string in docs
  [mln]
  (.getMaskArray mln))

(defn get-n-layers
  "get the number of layers in the mln"
  [mln]
  (.getnLayers mln))

(defn get-optimizer
  "returns this models optimizer"
  [mln]
  (.getOptimizer mln))

(defn get-output-layer
  "returns the output layer of the mln"
  [mln]
  (.getOutputLayer mln))

(defn get-param
  "get the parameter in question, param is the name of the param (string)"
  [mln param]
  (.getParam mln param))

(defn get-updater
  "return the updater used in this mln"
  [mln]
  (.getUpdater mln))

(defn gradient
  "calculate a gradient"
  [mln]
  (.gradient mln))

(defn get-gradient-and-score
  "get the gradient and score"
  [mln]
  (.gradientAndScore mln))

(defn init-model
  "initialize the model"
  [mln & {:keys [params clone-param-array?]
          :as opts}]
  (if (contains-many? opts :params :clone-param-array?)
    (.init mln params clone-param-array?)
    (.init mln))
  mln)

(defn init-gradients-view
  "initializes the flattened gradients array (used in backprop) and sets the appropriate subset in all layers."
  [mln]
  (doto mln
    (.initGradientsView)))

(defn initialize
  "Sets the input and labels from this dataset"
  [mln ds]
  (doto mln
    (.initialize ds)))

(defn initialize-layers
  "initialize the neuralNets based on the input."
  [mln input]
  (doto mln
    (.initializeLayers input)))

(defn init-params
  "initialize the parameters"
  [mln]
  (doto mln
    (.initParams)))

(defn get-feature-matrix
  "returns the input/feature matrix for the model"
  [mln]
  (.input mln))

(defn is-init-called?
  "was the model initialized"
  [mln]
  (.isInitCalled mln))

(defn is-pretrain-layer?
  "returns true if the layer can be trained in an unsupervised/pretain manner
   ie. VAE, RBMs"
  [mln]
  (.isPretrainLayer mln))

(defn run-one-iteration
  "runs one interation given the supplied input and mln"
  [mln input]
  (doto mln
    (.iterate input)))

(defn get-label-probabilities
  "returns the probabilities for each label for each example row wise"
  [mln examples]
  (.labelProbabilities mln examples))

(defn n-labels
  "returns the number of possible labels"
  [mln]
  (.numLabels mln))

(defn n-params
  "returns a 1 x m vector where the vector is composed of a flattened vector
  of all of the weights for the various neuralNets and output layer"
  [mln & {:keys [backwards?]
          :as opts}]
  (if (contains? opts :backwards?)
    (.numParams mln backwards?)
    (.numParams mln)))

(defn output
  "label the probabilities of the input or if masks are supplied,
  calculate the output of the network with masking arrays"
  [mln & {:keys [iterator train? input features-mask labels-mask
                 training-mode]
          :as opts}]
  (cond (contains-many? opts :input :train?
                        :features-mask :labels-mask)
        (.output mln input train? features-mask labels-mask)
        (contains-many? opts :training-mode :input)
        (.output mln input training-mode)
        (contains-many? opts :train? :input)
        (.output mln input train?)
        (contains-many? opts :iterator :train?)
        (.output mln iterator train?)
        (contains? opts :input)
        (.output mln input)
        (contains? mln :iterator)
        (.output mln iterator)
        :else
        (assert false "you must supply atleast an input or iterator")))

(defn params
  "Returns a 1 x m vector where the vector is composed of a flattened vector
  of all of the weights for the various neuralNets(w,hbias NOT VBIAS)
  and output layer"
  [mln & {:keys [backward-only?]
          :as opts}]
  (if (contains? opts :backward-only?)
    (.params mln backward-only?)
    (.params mln)))

(defn param-table
  "only mln supplied: The param table
  mln and backward-only? supplied:
  Table of parameters by key, for backprop For many models (dense layers, etc)
  - all parameters are backprop parameters"
  [mln & {:keys [backward-only?]
          :as opts}]
  (if (contains? opts :backward-only)
    (.paramTable mln backward-only?)
    (.paramTable mln)))

(defn predict
  "if you supply a dataset, returns the predicted label names
  if you supply an array of examples, returns the predictions for each example in the array"
  [mln {:keys [ds data]
        :as opts}]
  (cond (contains? opts :ds)
        (.predict mln ds)
        (contains? opts :data)
        (.predict mln data)
        :else
        (assert false "you must supply either a dataset or an array of examples.")))

(defn pre-output
  "returns the raw activations"
  [mln & {:keys [data training? training-mode]
          :as opts}]
  (cond (contains-many? opts :data :training-mode)
        (.preOutput mln data training-mode)
        (contains-many? opts :data :training?)
        (.preOutput mln data training?)
        (contains? opts :data)
        (.preOutput mln data)
        :else
        (assert false "you need data to generate activations")))

(defn pre-train!
  "Perform layerwise pretraining on all pre-trainable layers in the network (VAEs, RBMs, Autoencoders, etc)
  Note that pretraining will be performed on one layer after the other, resetting the DataSetIterator between iterations.
  For multiple epochs per layer, appropriately wrap the iterator (for example, a MultipleEpochsIterator)
  or train each layer manually using (pre-train-layer! layer-idx DataSetIterator)"
  [mln iterator]
  (.pretrain mln iterator))

(defn pre-train-layer!
  "Perform layerwise unsupervised training on a single pre-trainable layer
  in the network (VAEs, RBMs, Autoencoders, etc) If the specified layer index
  (0 to n-layers - 1) is not a pretrainable layer, this is a no-op."
  [mln & {:keys [layer-idx iterator features]
          :as opts}]
  (cond (contains-many? opts :layer-idx :iterator)
        (.pretrainLayer mln layer-idx iterator)
        (contains-many? opts :layer-idx :features)
        (.pretrainLayer mln layer-idx features)
        :else
        (assert false "you must supply the layer's index and either a dataset
 iterator or an array of features to pretrain on")))

(defn print-config
  "Prints the configuration and returns the mln"
  [mln]
  (doto mln
    (.printConfiguration)))

(defn reconstruct
  "reconstructs the input from the output of a given layer"
  [mln & {:keys [layer-output layer-idx]
          :as opts}]
  (if (contains-many? opts :layer-output :layer-idx)
    (.reconstruct mln layer-output layer-idx)
    (assert false "you must supply a layer and its output from the mln")))

(defn rnn-activate-using-stroed-state
  "returns the activation of the rnn given its most recent state"
  [mln & {:keys [input training? store-last-for-tbptt?]
          :as opts}]
  (if (contains-many? opts :input :training? :store-last-for-tbptt?)
    (.rnnActivateUsingStoredState mln input training? store-last-for-tbptt?)
    (assert false "you must supply a mln, the input to the model, if this is during training
or evaluation and if we want to store the previous state for truncated backprop")))

(defn rnn-clear-prev-state!
  "clear the previous state of the rnn layers if any and return the mln"
  [mln]
  (doto mln
    (.rnnClearPreviousState)))

(defn rnn-get-prev-state
  "get the state of the rnn layer given its index in the mln"
  [mln layer-idx]
  (.rnnGetPreviousState mln layer-idx))

(defn rnn-set-prev-state!
  "Set the state of the RNN layer and return the updated mln"
  [mln {:keys [layer-idx state]
        :as opts}]
  (cond (and (contains-many? opts :layer-idx :state)
             (map? state))
        (doto mln
          (.rnnSetPreviousState layer-idx state))
        :else
        (assert false "you must supply a layer-index for the layer in question within the mln and
a map of the desired state")))

(defn rnn-time-step
  "If this MultiLayerNetwork contains one or more RNN layers:
  conduct forward pass (prediction) but using previous stored state for any RNN layers."
  [mln input]
  (.rnnTimeStep mln input))

(defn score
  "only mln supplied: Score of the model (relative to the objective function)

  mln and dataset supplied: Sets the input and labels and returns a score for
   the prediction with respect to the true labels

  mln, dataset and training? supplied: Calculate the score (loss function)
  of the prediction with respect to the true labels"
  [mln & {:keys [dataset training?]
          :as opts}]
  (cond (contains-many? opts :dataset :training?)
        (.score mln dataset training?)
        (contains? opts :dataset)
        (.score mln dataset)
        :else
        (.score mln)))

(defn score-examples
  "Calculate the score for each example in a DataSet individually."
  [mln & {:keys [dataset add-regularization-terms? iterator]
          :as opts}]
  (cond (contains-many? opts :dataset :add-regularization-terms?)
        (.scoreExamples mln dataset add-regularization-terms?)
        (contains-many? opts :iterator :add-regularization-terms?)
        (.scoreExamples mln iterator add-regularization-terms?)
        :else
        (assert false "you must supply data in the form of a dataset or a dataset iterator.
you must also supply whether or not you want to add regularization terms (L1, L2, dropout...)")))

(defn set-conf!
  "Setter for the configuration, returns the mln"
  [mln nn-conf]
  (doto mln
    (.setConf nn-conf)))

(defn set-index!
  "Set the layer index and return the mln."
  [mln layer-idx]
  (doto mln
    (.setIndex layer-idx)))

(defn set-input!
  "Note that if input isn't nil and the neuralNets are nil,
  this is a way of initializing the neural network, returns the mln"
  [mln input]
  (doto mln
    (.setInput input)))

(defn set-input-mini-batch-size!
  "Set current/last input mini-batch size.
  Used for score and gradient calculations.
  returns the mln"
  [mln size]
  (doto mln
    (.setInputMiniBatchSize size)))

(defn set-labels!
  "sets the labels given an array of labels,
  returns the mln."
  [mln labels]
  (doto mln
    (.setLabels labels)))

(defn set-layer-mask-arrays!
  "Set the mask arrays for features and labels.
  returns the mln"
  [mln & {:keys [features-mask-array labels-mask-array]
          :as opts}]
  (cond (contains-many? opts :features-mask-array :labels-mask-array)
        (doto mln
          (.setLayerMaskArrays mln features-mask-array
                               labels-mask-array))
        :else
        (assert false "you need to supply a mask for the features and labels")))

(defn set-layers!
  "sets the layers of the mln in the order in which they appear in the supplied array.
  returns the mln"
  [mln layers]
  (doto mln
    (.setLayers layers)))

(defn set-layer-wise-config!
  "sets the configuration for a mln given a multi-layer configuration.
  returns the mln"
  [mln multi-layer-conf]
  (doto mln
    (.setLayerWiseConfigurations multi-layer-conf)))

(defn set-mask!
  "set the mask, returns the mln"
  [mln mask]
  (doto mln
    (.setMask mask)))

(defn set-mask-array!
  "set the mask array.
  returns the mln"
  [mln mask-array]
  (doto mln
    (.setMaskArray mask-array)))

(defn set-param!
  "Set the parameter with a new ndarray. returns the mln"
  [mln param]
  (doto mln
    (.setParam param)))

(defn set-params!
  "set the paramters for this model (mln).
  returns the mln"
  [mln params]
  (doto mln
    (.setParams params)))

(defn set-param-table!
  "setter for the param table. returns the mln"
  [mln param-map]
  (doto mln
    (.setParamTable param-map)))

(defn set-score!
  "sets the score (double), returns the mln"
  [mln score]
  (doto mln
    (.setScore score)))

(defn set-updater!
  "sets the updater for a given mln. returns the mln"
  [mln updater]
  (doto mln
    (.setUpdater updater)))

(defn summary
  "String detailing the architecture of the multilayernetwork. (mln)"
  [mln]
  (.summary mln))

(defn transpose!
  ;; I think this needs to be in the layer api ns
  "Return a transposed copy of the weights/bias
  (this means reverse the number of inputs and outputs on the weights)"
  [mln]
  (.transpose mln))

(defn layer-type
  ;; I think this needs to be in the layer api ns
  "returns the layer type"
  [layer]
  (.type layer))

(defn update!
  "given a gradient: Update layer weights and biases with gradient change

  given a gradient array and a param-type: Perform one update applying the gradient

  given another mln, Assigns the parameters of this model to the ones specified by this network.

  no matter the opts supplied, the mln is returned."
  [mln & {:keys [gradient gradient-array param-type other-mln]
          :as opts}]
  (cond (contains-many? opts :gradient-array :param-type)
        (doto mln
          (.update gradient-array param-type))
        (contains? opts :gradient)
        (doto mln
          (.update gradient))
        (contains? opts :other-mln)
        (doto mln
          (.update other-mln))
        :else
        (assert false "you must supply either a gradient, a gradient-array and param-type or
another mln")))

(defn update-rnn-state-with-tbptt-state!
  "updates the rnn state to be that of the tbptt state.
  returns the mln."
  [mln]
  (doto mln
    (.updateRnnStateWithTBPTTState)))

(defn validate-input!
  "validate the input and returns the mln"
  [mln]
  (doto mln
    (.validateInput)))

(defn z-from-prev-layer
  "Compute input linear transformation (z) from previous layer
  Applies pre processing transformation where necessary"
  [mln & {:keys [current-layer-idx input training?]
          :as opts}]
  (if (contains-many? opts :current-layer-idx :input :training?)
    (.zFromPrevLayer mln current-layer-idx input training?)
    (assert false "you must supply the index of the current layer, an input array and if this is for training or evaluation")))
