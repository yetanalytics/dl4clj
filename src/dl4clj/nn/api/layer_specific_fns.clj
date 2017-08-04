(ns ^{:doc "see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/Layer.html
and https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/package-frame.html"}
    dl4clj.nn.api.layer-specific-fns
  (:import [org.deeplearning4j.nn.conf.layers BaseOutputLayer
            CenterLossOutputLayer SubsamplingLayer]
           [org.deeplearning4j.nn.layers
            BasePretrainNetwork FrozenLayer LossLayer]
           [org.deeplearning4j.nn.layers.feedforward.autoencoder AutoEncoder]
           [org.deeplearning4j.nn.layers.feedforward.rbm RBM]
           [org.deeplearning4j.nn.layers.normalization BatchNormalization]
           [org.deeplearning4j.nn.layers.variational VariationalAutoencoder]
           [org.deeplearning4j.nn.api.layers RecurrentLayer]
           [org.deeplearning4j.nn.api.layers IOutputLayer])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [dl4clj.utils :refer [contains-many?]]
            [clojure.core.match :refer [match]]
            [dl4clj.constants :as enum]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From BaseOutputlayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-loss-fn
  [output-layer]
  (.getLossFn output-layer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From CenterLossOutputLayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-alpha
  [center-loss-output-layer]
  (.getAlpha center-loss-output-layer))

(defn get-gradient-check
  [center-loss-output-layer]
  (.getGradientCheck center-loss-output-layer))

(defn get-lambda
  [center-loss-output-layer]
  (.getLambda center-loss-output-layer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From SubsamplingLayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-eps
  [subsampling-layer]
  (.getEps subsampling-layer))

(defn get-pnorm
  [subsampling-layer]
  (.getPnorm subsampling-layer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; output layers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn compute-score
  "Compute score after labels and input have been set.

  output-layer is the output layer in question
  full-network-l1 (double) is the l1 regularization term for the model the layer is apart of
  full-network-l2 (double) is the l2 regularization term for the model the layer is aprt of
   -note: it is okay for L1 and L2 to be set to 0.0 if regularization was not used
  training? (boolean) are we traing the model or testing it?"
  [& {:keys [output-layer full-network-l1 full-network-l2 training?]}]
  (.computeScore output-layer full-network-l1 full-network-l2 training?))

(defn compute-score-for-examples
  "Compute the score for each example individually, after labels and input have been set.

  output-layer is the output layer in question
  full-network-l1 (double) is the l1 regularization term for the model the layer is apart of
  full-network-l2 (double) is the l2 regularization term for the model the layer is aprt of
   -note: it is okay for L1 and L2 to be set to 0.0 if regularization was not used"
  [& {:keys [output-layer full-network-l1 full-network-l2]}]
  (.computeScoreForExamples output-layer full-network-l1 full-network-l2))

(defn set-labels!
  "Set the labels array for this output layer and returns the layer

  labels is an INDArray or vec of labels"
  [& {:keys [output-layer labels]}]
  (doto output-layer
    (.setLabels (vec-or-matrix->indarray labels))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; base pretrain layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-corrupted-input
  "Corrupts the given input by doing a binomial sampling given the corruption level

  :features is an INDArray or vector of input features to the network

  :corruption-level (double) amount of corruption to apply."
  [& {:keys [base-pretrain-network features corruption-level]}]
  (.getCorruptedInput base-pretrain-network (vec-or-matrix->indarray features)
                      corruption-level))

(defn sample-hidden-given-visible
  "Sample the hidden distribution given the visible

  :visible is an INDArray or vector of the visibile distribution"
  [& {:keys [base-pretrain-network visible]}]
  (.sampleHiddenGivenVisible base-pretrain-network
                             (vec-or-matrix->indarray visible)))

(defn sample-visible-given-hidden
  "Sample the visible distribution given the hidden

  :hidden is an INDArray or vector of the hidden distribution"
  [& {:keys [base-pretrain-network hidden]}]
  (.sampleVisibleGivenHidden base-pretrain-network
                             (vec-or-matrix->indarray hidden)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; autoencoders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn decode
  "decodes an encoded output of an autoencoder"
  [& {:keys [autoencoder layer-output]}]
  (.decode autoencoder (vec-or-matrix->indarray layer-output)))

(defn encode
  "encodes an input to be passed to an autoencoder"
  [& {:keys [autoencoder input training?]}]
  (.encode autoencoder (vec-or-matrix->indarray input) training?))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rbm
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn gibbs-sampling-step
  "Gibbs sampling step: hidden ---> visible ---> hidden
  returns the expected values and samples of both the visible samples given
  the hidden and the new hidden input and expected values"
  [& {:keys [rbm hidden-input]}]
  (.gibbhVh rbm (vec-or-matrix->indarray hidden-input)))

(defn prop-up
  "Calculates the activation of the visible : sigmoid(v * W + hbias)"
  [& {:keys [rbm visible-input training?]
      :as opts}]
  (assert (contains-many? opts :rbm :visible-input)
          "you must supply a rbm and the visible input")
  (let [vi (vec-or-matrix->indarray visible-input)]
   (if training?
    (.propUp rbm vi training?)
    (.propUp rbm vi))))

(defn prop-up-derivative
  "derivative of the prop-up activation"
  [& {:keys [rbm prop-up-vals]}]
  (.propUpDerivative rbm (vec-or-matrix->indarray prop-up-vals)))

(defn prop-down
  "Calculates the activation of the hidden: (activation (h * W + vbias))"
  [& {:keys [rbm hidden-input]}]
  (.propDown rbm (vec-or-matrix->indarray hidden-input)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; frozen layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn log-test-mode
  ;; figure out exactly what this does. no docs for this method
  ":training? (boolean) training or testing?

  :training-mode (keyword) one of :training or :testing"
  [& {:keys [frozen-layer training? training-mode]
      :as opts}]
  (match [opts]
         [{:frozen-layer _ :training? _}]
         (doto frozen-layer (.logTestMode training?))
         [{:frozen-layer _ :training-mode _}]
         (doto frozen-layer
           (.logTestMode (enum/value-of {:layer-training-mode training-mode})))
         :else
         (assert false "you must supply training? or training-mode")))

(defn get-inside-layer
  [frozen-layer]
  (.getInsideLayer frozen-layer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; batch normalization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-shape
  "returns an integer array of the shape after passing through the normalization layer"
  [& {:keys [batch-norm features]}]
  (.getShape batch-norm (vec-or-matrix->indarray features)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; vae
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn is-pretrain-param?
  "checks to see if the supplied param can be pretrained

  :param (string), name of the paramter"
  [& {:keys [vae param]}]
  (.isPretrainParam vae param))

(defn reconstruction-probability
  "Calculate the reconstruction probability,
   as described in An & Cho, 2015
    - Variational Autoencoder based Anomaly Detection using Reconstruction Probability
      - (Algorithm 4)
  The authors describe it as follows:
   - This is essentially the probability of the data being generated from a given
     latent variable drawn from the approximate posterior distribution.

  Specifically, for each example x in the input, calculate p(x). Note however that p(x)
  is a stochastic (Monte-Carlo) estimate of the true p(x), based on the specified
  number of samples. More samples will produce a more accurate (lower variance)
  estimate of the true p(x) for the current model parameters.


  The returned array is a column vector of reconstruction probabilities,
  for each example. Thus, reconstruction probabilities can (and should, for efficiency)
  be calculated in a batched manner.

  under the hood, this fn calls reconstructionLogProbability in dl4j land.

  :data (INDArray or vec), the data to calculate the reconstruction probability for

  :num-samples (int), Number of samples with which to base the reconstruction probability on."
  [& {:keys [vae data num-samples]}]
  (.reconstructionProbability vae (vec-or-matrix->indarray data) num-samples))

(defn reconstruction-log-probability
  "Return the log reconstruction probability given the specified number of samples.

  the docs in reconstruction-probability also apply to this fn."
  [& {:keys [vae data num-samples]}]
  (.reconstructionLogProbability vae (vec-or-matrix->indarray data) num-samples))

(defn generate-random-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  randomly generate output x, where x ~ P(x|z)

  :latent-space-values (INDArray or vec), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values]}]
  (.generateRandomGivenZ vae (vec-or-matrix->indarray latent-space-values)))

(defn generate-at-mean-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  generate output from P(x|z), where x = E[P(x|z)]
   - i.e., return the mean value for the distribution P(x|z)

  :latent-space-values (INDArray or vec), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values]}]
  (.generateAtMeanGivenZ vae (vec-or-matrix->indarray latent-space-values)))

(defn has-loss-fn?
  "Does the reconstruction distribution have a loss function (such as mean squared error)
  or is it a standard probabilistic reconstruction distribution?"
  [vae]
  (.hasLossFunction vae))

(defn reconstruction-error
  "Return the reconstruction error for this variational autoencoder.

  This method is used ONLY for VAEs that have a standard neural network loss function
  (i.e., an ILossFunction instance such as mean squared error) instead of using
  a probabilistic reconstruction distribution P(x|z) for the reconstructions
  (as presented in the VAE architecture by Kingma and Welling).

  reconstruction error is a simple deterministic function

  :vae (layer), is the variational autoencoder

  :data (INDArray or vec), the data to calc the reconstruction error on"
  [& {:keys [vae data]}]
  (.reconstructionError vae (vec-or-matrix->indarray data)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; recurrent layers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn rnn-activate-using-stored-state
  "Similar to rnnTimeStep, this method is used for activations using the state
  stored in the stateMap as the initialization.

  input (INDArray or vec) of input values to the layer
  store-last-for-tbptt? (boolean), save state to be used in tbptt
  training? (boolean) is the model currently in training or not?"
  [& {:keys [rnn-layer input training? store-last-for-tbptt?]}]
  (.rnnActivateUsingStoredState rnn-layer (vec-or-matrix->indarray input)
                                training? store-last-for-tbptt?))

(defn rnn-clear-previous-state!
  "Reset/clear the stateMap for rnn-time-step and
  tBpttStateMap for rnn-activate-using-stored-state

  returns the rnn-layer"
  [rnn-layer]
  (doto rnn-layer
    (.rnnClearPreviousState)))

(defn rnn-layer-get-prev-state
  "Returns a shallow copy of the RNN stateMap (that contains the stored history
  for use in fns such as rnn-time-step"
  [rnn-layer]
  (.rnnGetPreviousState rnn-layer))

(defn rnn-get-tbptt-state
  "Get the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer."
  [rnn-layer]
  (.rnnGetTBPTTState rnn-layer))

(defn rnn-set-tbptt-state!
  "Set the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer.
  and returns the layer

  state is a map of {string indArray}"
  [& {:keys [rnn-layer state]}]
  (doto rnn-layer
    (.rnnSetTBPTTState state)))

(defn rnn-layer-set-prev-state!
  "Set the stateMap (stored history) and return the layer.

  state is a map of {string indArray}"
  [& {:keys [rnn-layer state]}]
  (doto rnn-layer
    (.rnnSetPreviousState state)))

(defn rnn-layer-time-step
  "Do one or more time steps using the previous time step state stored in stateMap.
  Can be used to efficiently do forward pass one or n-steps at a time (instead of doing forward pass always from t=0)
  If stateMap is empty, default initialization (usually zeros) is used
  Implementations also update stateMap at the end of this method

  input should be an INDArray or vector of input values to the layer"
  [& {:keys [rnn-layer input]}]
  (.rnnTimeStep rnn-layer (vec-or-matrix->indarray input)))

(defn tbptt-backprop-gradient
  "Returns the Truncated BPTT gradient

  epsilon should be an INDArray or vec
  tbptt-back-length is an integer"
  [& {:keys [rnn-layer epsilon tbptt-back-length]}]
  (.tbpttBackpropGradient rnn-layer (vec-or-matrix->indarray epsilon)
                          tbptt-back-length))
