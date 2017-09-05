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
  (match [output-layer]
         [(_ :guard seq?)]
         `(.getLossFn ~output-layer)
         :else
         (.getLossFn output-layer)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From CenterLossOutputLayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-alpha
  [center-loss-output-layer]
  (match [center-loss-output-layer]
         [(_ :guard seq?)]
         `(.getAlpha ~center-loss-output-layer)
         :else
         (.getAlpha center-loss-output-layer)))

(defn get-gradient-check
  [center-loss-output-layer]
  (match [center-loss-output-layer]
         [(_ :guard seq?)]
         `(.getGradientCheck ~center-loss-output-layer)
         :else
         (.getGradientCheck center-loss-output-layer)))

(defn get-lambda
  [center-loss-output-layer]
  (match [center-loss-output-layer]
         [(_ :guard seq?)]
         `(.getLambda ~center-loss-output-layer)
         :else
         (.getLambda center-loss-output-layer)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From SubsamplingLayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-eps
  [subsampling-layer]
  (match [subsampling-layer]
         [(_ :guard seq?)]
         `(.getEps ~subsampling-layer)
         :else
         (.getEps subsampling-layer)))

(defn get-pnorm
  [subsampling-layer]
  (match [subsampling-layer]
         [(_ :guard seq?)]
         `(.getPnorm ~subsampling-layer)
         :else
         (.getPnorm subsampling-layer)))

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
  [& {:keys [output-layer full-network-l1 full-network-l2 training?]
      :as opts}]
  (match [opts]
         [{:output-layer (_ :guard seq?)
           :full-network-l1 (:or (_ :guard number?)
                                 (_ :guard seq?))
           :full-network-l2 (:or (_ :guard number?)
                                 (_ :guard seq?))
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         `(.computeScore ~output-layer (double ~full-network-l1)
                         (double ~full-network-l2) ~training?)
         :else
         (.computeScore output-layer full-network-l1 full-network-l2 training?)))

(defn compute-score-for-examples
  "Compute the score for each example individually, after labels and input have been set.

  output-layer is the output layer in question
  full-network-l1 (double) is the l1 regularization term for the model the layer is apart of
  full-network-l2 (double) is the l2 regularization term for the model the layer is aprt of
   -note: it is okay for L1 and L2 to be set to 0.0 if regularization was not used"
  [& {:keys [output-layer full-network-l1 full-network-l2]
      :as opts}]
  (match [opts]
         [{:output-layer (_ :guard seq?)
           :full-network-l1 (:or (_ :guard number?)
                                 (_ :guard seq?))
           :full-network-l2 (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         `(.computeScoreForExamples ~output-layer (double ~full-network-l1)
                                    (double ~full-network-l2))
         :else
         (.computeScoreForExamples output-layer full-network-l1 full-network-l2)))

(defn set-labels!
  "Set the labels array for this output layer and returns the layer

  labels is an INDArray or vec of labels"
  [& {:keys [output-layer labels]
      :as opts}]
  (match [opts]
         [{:output-layer (_ :guard seq?)
           :labels (:or (_ :guard seq?)
                        (_ :guard vector?))}]
         `(doto ~output-layer
            (.setLabels (vec-or-matrix->indarray ~labels)))
         :else
         (doto output-layer
           (.setLabels (vec-or-matrix->indarray labels)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; base pretrain layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-corrupted-input
  "Corrupts the given input by doing a binomial sampling given the corruption level

  :features is an INDArray or vector of input features to the network

  :corruption-level (double) amount of corruption to apply."
  [& {:keys [base-pretrain-network features corruption-level]
      :as opts}]
  (match [opts]
         [{:base-pretrain-network (_ :guard seq?)
           :features (:or (_ :guard seq?)
                          (_ :guard vector?))
           :corruption-level (:or (_ :guard number?)
                                  (_ :guard seq?))}]
         `(.getCorruptedInput ~base-pretrain-network (vec-or-matrix->indarray ~features)
                              (double ~corruption-level))
         :else
         (.getCorruptedInput base-pretrain-network (vec-or-matrix->indarray features)
                             corruption-level)))

(defn sample-hidden-given-visible
  "Sample the hidden distribution given the visible

  :visible is an INDArray or vector of the visibile distribution"
  [& {:keys [base-pretrain-network visible]
      :as opts}]
  (match [opts]
         [{:base-pretrain-network (_ :guard seq?)
           :visible (:or (_ :guard seq?)
                         (_ :guard vector?))}]
         `(.sampleHiddenGivenVisible ~base-pretrain-network
                                     (vec-or-matrix->indarray ~visible))
         :else
         (.sampleHiddenGivenVisible base-pretrain-network
                                    (vec-or-matrix->indarray visible))))

(defn sample-visible-given-hidden
  "Sample the visible distribution given the hidden

  :hidden is an INDArray or vector of the hidden distribution"
  [& {:keys [base-pretrain-network hidden]
      :as opts}]
  (match [opts]
         [{:base-pretrain-network (_ :guard seq?)
           :hidden (:or (_ :guard seq?)
                        (_ :guard vector?))}]
         `(.sampleVisibleGivenHidden ~base-pretrain-network
                                     (vec-or-matrix->indarray ~hidden))
         :else
         (.sampleVisibleGivenHidden base-pretrain-network
                                    (vec-or-matrix->indarray hidden))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; autoencoders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn decode
  "decodes an encoded output of an autoencoder"
  [& {:keys [autoencoder layer-output]
      :as opts}]
  (match [opts]
         [{:autoencoder (_ :guard seq?)
           :layer-output (:or (_ :guard seq?)
                              (_ :guard vector?))}]
         `(.decode ~autoencoder (vec-or-matrix->indarray ~layer-output))
         :else
         (.decode autoencoder (vec-or-matrix->indarray layer-output))))

(defn encode
  "encodes an input to be passed to an autoencoder"
  [& {:keys [autoencoder input training?]
      :as opts}]
  (match [opts]
         [{:autoencoder (_ :guard seq?)
           :input (:or (_ :guard seq?)
                       (_ :guard vector?))
           :training (:or (_ :guard boolean?)
                          (_ :guard seq?))}]
         `(.encode ~autoencoder (vec-or-matrix->indarray ~input) ~training?)
         :else
         (.encode autoencoder (vec-or-matrix->indarray input) training?)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rbm
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn gibbs-sampling-step
  "Gibbs sampling step: hidden ---> visible ---> hidden
  returns the expected values and samples of both the visible samples given
  the hidden and the new hidden input and expected values"
  [& {:keys [rbm hidden-input]
      :as opts}]
  (match [opts]
         [{:rbm (_ :guard seq?)
           :hidden-input (:or (_ :guard seq?)
                              (_ :guard vector?))}]
         `(.gibbhVh ~rbm (vec-or-matrix->indarray ~hidden-input))
         :else
         (.gibbhVh rbm (vec-or-matrix->indarray hidden-input))))

(defn prop-up
  "Calculates the activation of the visible : sigmoid(v * W + hbias)"
  [& {:keys [rbm visible-input training?]
      :as opts}]
  (assert (contains-many? opts :rbm :visible-input)
          "you must supply a rbm and the visible input")
  (match [opts]
         [{:rbm (_ :guard seq?)
           :visible-input (:or (_ :guard seq?)
                               (_ :guard vector?))
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         `(.propUp ~rbm (vec-or-matrix->indarray ~visible-input) ~training?)
         [{:rbm _
           :visible-input _
           :training? _}]
         (.propUp rbm (vec-or-matrix->indarray visible-input) training?)
         [{:rbm (_ :guard seq?)
           :visible-input (:or (_ :guard seq?)
                               (_ :guard vector?))}]
         `(.propUp ~rbm (vec-or-matrix->indarray ~visible-input))
         :else
         (.propUp rbm (vec-or-matrix->indarray visible-input))))

(defn prop-up-derivative
  "derivative of the prop-up activation"
  [& {:keys [rbm prop-up-vals]
      :as opts}]
  (match [opts]
         [{:rbm (_ :guard seq?)
           :prop-up-vals (:or (_ :guard seq?)
                              (_ :guard vector?))}]
         `(.propUpDerivative ~rbm (vec-or-matrix->indarray ~prop-up-vals))
         :else
         (.propUpDerivative rbm (vec-or-matrix->indarray prop-up-vals))))

(defn prop-down
  "Calculates the activation of the hidden: (activation (h * W + vbias))"
  [& {:keys [rbm hidden-input]
      :as opts}]
  (match [opts]
         [{:rbm (_ :guard seq?)
           :hidden-input (:or (_ :guard seq?)
                              (_ :guard vector?))}]
         `(.propDown ~rbm (vec-or-matrix->indarray ~hidden-input))
         :else
         (.propDown rbm (vec-or-matrix->indarray hidden-input))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; frozen layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn log-test-mode
  ":training? (boolean) training or testing?

  :training-mode (keyword) one of :training or :testing"
  [& {:keys [frozen-layer training? training-mode]
      :as opts}]
  (match [opts]
         [{:frozen-layer (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         `(doto ~frozen-layer (.logTestMode ~training?))
         [{:frozen-layer _ :training? _}]
         (doto frozen-layer (.logTestMode training?))
         [{:frozen-layer (_ :guard seq?)
           :training-mode (:or (_ :guard keyword?)
                               (_ :guard seq?))}]
         `(doto ~frozen-layer
            (.logTestMode (enum/value-of {:layer-training-mode ~training-mode})))
         [{:frozen-layer _ :training-mode _}]
         (doto frozen-layer
           (.logTestMode (enum/value-of {:layer-training-mode training-mode})))))

(defn get-inside-layer
  [frozen-layer]
  (match [frozen-layer]
         [(_ :guard seq?)]
         `(.getInsideLayer ~frozen-layer)
         :else
         (.getInsideLayer frozen-layer)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; batch normalization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-shape
  "returns an integer array of the shape after passing through the normalization layer"
  [& {:keys [batch-norm features]
      :as opts}]
  (match [opts]
         [{:batch-norm (_ :guard seq?)
           :features (:or (_ :guard seq?)
                          (_ :guard vector?))}]
         `(.getShape ~batch-norm (vec-or-matrix->indarray ~features))
         :else
         (.getShape batch-norm (vec-or-matrix->indarray features))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; vae
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn is-pretrain-param?
  "checks to see if the supplied param can be pretrained

  :param (string), name of the paramter"
  [& {:keys [vae param]
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :param (:or (_ :guard string?)
                       (_ :guard seq?))}]
         `(.isPretrainParam ~vae ~param)
         :else
         (.isPretrainParam vae param)))

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
  [& {:keys [vae data num-samples]
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :data (:or (_ :guard seq?)
                      (_ :guard vector?))
           :num-samples (:or (_ :guard number?)
                             (_ :guard seq?))}]
         `(.reconstructionProbability ~vae (vec-or-matrix->indarray ~data) (int ~num-samples))
         :else
         (.reconstructionProbability vae (vec-or-matrix->indarray data) num-samples)))

(defn reconstruction-log-probability
  "Return the log reconstruction probability given the specified number of samples.

  the docs in reconstruction-probability also apply to this fn."
  [& {:keys [vae data num-samples]
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :data (:or (_ :guard seq?)
                      (_ :guard vector?))
           :num-samples (:or (_ :guard number?)
                             (_ :guard seq?))}]
         `(.reconstructionLogProbability ~vae (vec-or-matrix->indarray ~data)
                                         (int ~num-samples))
         :else
         (.reconstructionLogProbability vae (vec-or-matrix->indarray data) num-samples)))

(defn generate-random-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  randomly generate output x, where x ~ P(x|z)

  :latent-space-values (INDArray or vec), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values]
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :latent-space-values (:or (_ :guard seq?)
                                     (_ :guard vector?))}]
         `(.generateRandomGivenZ ~vae (vec-or-matrix->indarray ~latent-space-values))
         :else
         (.generateRandomGivenZ vae (vec-or-matrix->indarray latent-space-values))))

(defn generate-at-mean-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  generate output from P(x|z), where x = E[P(x|z)]
   - i.e., return the mean value for the distribution P(x|z)

  :latent-space-values (INDArray or vec), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values]
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :latent-space-values (:or (_ :guard seq?)
                                     (_ :guard vector?))}]
         `(.generateAtMeanGivenZ ~vae (vec-or-matrix->indarray ~latent-space-values))
         :else
         (.generateAtMeanGivenZ vae (vec-or-matrix->indarray latent-space-values))))

(defn has-loss-fn?
  "Does the reconstruction distribution have a loss function (such as mean squared error)
  or is it a standard probabilistic reconstruction distribution?"
  [vae]
  (match [vae]
         [(_ :guard seq?)]
         `(.hasLossFunction ~vae)
         :else
         (.hasLossFunction vae)))

(defn reconstruction-error
  "Return the reconstruction error for this variational autoencoder.

  This method is used ONLY for VAEs that have a standard neural network loss function
  (i.e., an ILossFunction instance such as mean squared error) instead of using
  a probabilistic reconstruction distribution P(x|z) for the reconstructions
  (as presented in the VAE architecture by Kingma and Welling).

  reconstruction error is a simple deterministic function

  :vae (layer), is the variational autoencoder

  :data (INDArray or vec), the data to calc the reconstruction error on"
  [& {:keys [vae data]
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :data (:or (_ :guard seq?)
                      (_ :guard vector?))}]
         `(.reconstructionError ~vae (vec-or-matrix->indarray ~data))
         :else
         (.reconstructionError vae (vec-or-matrix->indarray data))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; recurrent layers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn rnn-activate-using-stored-state
  "Similar to rnnTimeStep, this method is used for activations using the state
  stored in the stateMap as the initialization.

  input (INDArray or vec) of input values to the layer
  store-last-for-tbptt? (boolean), save state to be used in tbptt
  training? (boolean) is the model currently in training or not?"
  [& {:keys [rnn-layer input training? store-last-for-tbptt?]
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :input (:or (_ :guard seq?)
                       (_ :guard vector?))
           :training? (:or (_ :guard seq?)
                           (_ :guard boolean?))
           :store-last-for-tbptt? (:or (_ :guard seq?)
                                       (_ :guard boolean?))}]
         `(.rnnActivateUsingStoredState rnn-layer (vec-or-matrix->indarray input)
                                        training? store-last-for-tbptt?)
         :else
         (.rnnActivateUsingStoredState rnn-layer (vec-or-matrix->indarray input)
                                       training? store-last-for-tbptt?)))

(defn rnn-clear-previous-state!
  "Reset/clear the stateMap for rnn-time-step and
  tBpttStateMap for rnn-activate-using-stored-state

  returns the rnn-layer"
  [rnn-layer]
  (match [rnn-layer]
         [(_ :guard seq?)]
         `(doto ~rnn-layer
            (.rnnClearPreviousState))
         :else
         (doto rnn-layer
           (.rnnClearPreviousState))))

(defn rnn-layer-get-prev-state
  "Returns a shallow copy of the RNN stateMap (that contains the stored history
  for use in fns such as rnn-time-step"
  [rnn-layer]
  (match [rnn-layer]
         [(_ :guard seq?)]
         `(.rnnGetPreviousState ~rnn-layer)
         :else
         (.rnnGetPreviousState rnn-layer)))

(defn rnn-get-tbptt-state
  "Get the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer."
  [rnn-layer]
  (match [rnn-layer]
         [(_ :guard seq?)]
         `(.rnnGetTBPTTState ~rnn-layer)
         :else
         (.rnnGetTBPTTState rnn-layer)))

(defn rnn-set-tbptt-state!
  "Set the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer.
  and returns the layer

  state is a map of {string (indArray or code for creating one)}"
  [& {:keys [rnn-layer state]
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :state (_ :guard map?)}]
         `(doto ~rnn-layer
            (.rnnSetTBPTTState ~state))
         :else
         (doto rnn-layer
           (.rnnSetTBPTTState state))))

(defn rnn-layer-set-prev-state!
  "Set the stateMap (stored history) and return the layer.

  state is a map of {string (indArray or code for creating one)}
   - where seq is the code for creating an indArray

  There is no validation for whats in the state map"
  [& {:keys [rnn-layer state]
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :state (_ :guard map?)}]
         `(doto ~rnn-layer
            (.rnnSetPreviousState ~state))
         :else
         (doto rnn-layer
           (.rnnSetPreviousState state))))

(defn rnn-layer-time-step
  "Do one or more time steps using the previous time step state stored in stateMap.
  Can be used to efficiently do forward pass one or n-steps at a time (instead of doing forward pass always from t=0)
  If stateMap is empty, default initialization (usually zeros) is used
  Implementations also update stateMap at the end of this method

  input should be an INDArray or vector of input values to the layer"
  [& {:keys [rnn-layer input]
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :input (:or (_ :guard seq?)
                       (_ :guard vector?))}]
         `(.rnnTimeStep ~rnn-layer (vec-or-matrix->indarray ~input))
         :else
         (.rnnTimeStep rnn-layer (vec-or-matrix->indarray input))))

(defn tbptt-backprop-gradient
  "Returns the Truncated BPTT gradient

  epsilon should be an INDArray or vec
  tbptt-back-length is an integer"
  [& {:keys [rnn-layer epsilon tbptt-back-length]
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :epsilon (:or (_ :guard seq?)
                         (_ :guard vector?))
           :tbptt-back-length (:or (_ :guard number?)
                                   (_ :guard seq?))}]
         `(.tbpttBackpropGradient ~rnn-layer (vec-or-matrix->indarray ~epsilon)
                                  (int ~tbptt-back-length))
         :else
         (.tbpttBackpropGradient rnn-layer (vec-or-matrix->indarray epsilon)
                                 tbptt-back-length)))
