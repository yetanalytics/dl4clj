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
            [dl4clj.utils :refer [contains-many? obj-or-code? eval-if-code]]
            [clojure.core.match :refer [match]]
            [dl4clj.constants :as enum]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From BaseOutputlayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-loss-fn
  [output-layer & {:keys [as-code?]
                   :or {as-code? true}}]
  (match [output-layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLossFn ~output-layer))
         :else
         (.getLossFn output-layer)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From CenterLossOutputLayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-alpha
  [center-loss-output-layer & {:keys [as-code?]
                               :or {as-code? true}}]
  (match [center-loss-output-layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getAlpha ~center-loss-output-layer))
         :else
         (.getAlpha center-loss-output-layer)))

(defn get-gradient-check
  [center-loss-output-layer & {:keys [as-code?]
                               :or {as-code? true}}]
  (match [center-loss-output-layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getGradientCheck ~center-loss-output-layer))
         :else
         (.getGradientCheck center-loss-output-layer)))

(defn get-lambda
  [center-loss-output-layer & {:keys [as-code?]
                               :or {as-code? true}}]
  (match [center-loss-output-layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLambda ~center-loss-output-layer))
         :else
         (.getLambda center-loss-output-layer)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From SubsamplingLayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-eps
  [subsampling-layer & {:keys [as-code?]
                        :or {as-code? true}}]
  (match [subsampling-layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getEps ~subsampling-layer))
         :else
         (.getEps subsampling-layer)))

(defn get-pnorm
  [subsampling-layer & {:keys [as-code?]
                        :or {as-code? true}}]
  (match [subsampling-layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getPnorm ~subsampling-layer))
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
  [& {:keys [output-layer full-network-l1 full-network-l2 training? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:output-layer (_ :guard seq?)
           :full-network-l1 (:or (_ :guard number?)
                                 (_ :guard seq?))
           :full-network-l2 (:or (_ :guard number?)
                                 (_ :guard seq?))
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.computeScore ~output-layer (double ~full-network-l1)
                         (double ~full-network-l2) ~training?))
         :else
         (let [[o-l n-l1 n-l2 t?]
               (eval-if-code [output-layer seq?]
                             [full-network-l1 seq? number?]
                             [full-network-l2 seq? number?]
                             [training? seq? boolean?])]
           (.computeScore o-l n-l1 n-l2 t?))))

(defn compute-score-for-examples
  "Compute the score for each example individually, after labels and input have been set.

  output-layer is the output layer in question
  full-network-l1 (double) is the l1 regularization term for the model the layer is apart of
  full-network-l2 (double) is the l2 regularization term for the model the layer is aprt of
   -note: it is okay for L1 and L2 to be set to 0.0 if regularization was not used"
  [& {:keys [output-layer full-network-l1 full-network-l2 as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:output-layer (_ :guard seq?)
           :full-network-l1 (:or (_ :guard number?)
                                 (_ :guard seq?))
           :full-network-l2 (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.computeScoreForExamples ~output-layer (double ~full-network-l1)
                                    (double ~full-network-l2)))
         :else
         (let [[n-l1 n-l2] (eval-if-code [full-network-l1 seq? number?]
                                         [full-network-l2 seq? number?])]
           (.computeScoreForExamples output-layer n-l1 n-l2))))

(defn set-labels!
  "Set the labels array for this output layer and returns the layer

  labels is an INDArray or vec of labels"
  [& {:keys [output-layer labels as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:output-layer (_ :guard seq?)
           :labels (:or (_ :guard seq?)
                        (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(doto ~output-layer
            (.setLabels (vec-or-matrix->indarray ~labels))))
         :else
         (let [[o-layer l] (eval-if-code [output-layer seq?] [labels seq?])]
          (doto o-layer (.setLabels (vec-or-matrix->indarray l))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; base pretrain layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-corrupted-input
  "Corrupts the given input by doing a binomial sampling given the corruption level

  :features is an INDArray or vector of input features to the network

  :corruption-level (double) amount of corruption to apply."
  [& {:keys [base-pretrain-network features corruption-level as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:base-pretrain-network (_ :guard seq?)
           :features (:or (_ :guard seq?)
                          (_ :guard vector?))
           :corruption-level (:or (_ :guard number?)
                                  (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.getCorruptedInput ~base-pretrain-network (vec-or-matrix->indarray ~features)
                              (double ~corruption-level)))
         :else
         (let [[f c-lvl bpn] (eval-if-code [features seq?]
                                           [corruption-level seq? number?]
                                           [base-pretrain-network seq?])]
           (.getCorruptedInput bpn (vec-or-matrix->indarray f) c-lvl))))

(defn sample-hidden-given-visible
  "Sample the hidden distribution given the visible

  :visible is an INDArray or vector of the visibile distribution"
  [& {:keys [base-pretrain-network visible as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:base-pretrain-network (_ :guard seq?)
           :visible (:or (_ :guard seq?)
                         (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(.sampleHiddenGivenVisible ~base-pretrain-network
                                     (vec-or-matrix->indarray ~visible)))
         :else
         (let [[v bpn] (eval-if-code [visible seq?] [base-pretrain-network seq?])]
          (.sampleHiddenGivenVisible bpn (vec-or-matrix->indarray v)))))

(defn sample-visible-given-hidden
  "Sample the visible distribution given the hidden

  :hidden is an INDArray or vector of the hidden distribution"
  [& {:keys [base-pretrain-network hidden as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:base-pretrain-network (_ :guard seq?)
           :hidden (:or (_ :guard seq?)
                        (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(.sampleVisibleGivenHidden ~base-pretrain-network
                                     (vec-or-matrix->indarray ~hidden)))
         :else
         (let [[h bpn] (eval-if-code [hidden seq?] [base-pretrain-network seq?])]
           (.sampleVisibleGivenHidden bpn (vec-or-matrix->indarray h)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; autoencoders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn decode
  "decodes an encoded output of an autoencoder"
  [& {:keys [autoencoder layer-output as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:autoencoder (_ :guard seq?)
           :layer-output (:or (_ :guard seq?)
                              (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(.decode ~autoencoder (vec-or-matrix->indarray ~layer-output)))
         :else
         (let [[l-o a] (eval-if-code [layer-output seq?] [autoencoder seq?])]
           (.decode a (vec-or-matrix->indarray l-o)))))

(defn encode
  "encodes an input to be passed to an autoencoder"
  [& {:keys [autoencoder input training? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:autoencoder (_ :guard seq?)
           :input (:or (_ :guard seq?)
                       (_ :guard vector?))
           :training (:or (_ :guard boolean?)
                          (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.encode ~autoencoder (vec-or-matrix->indarray ~input) ~training?))
         :else
         (let [[i t? a] (eval-if-code [input seq?]
                                      [training? seq? boolean?]
                                      [autoencoder seq?])]
           (.encode a (vec-or-matrix->indarray i) t?))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rbm
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn gibbs-sampling-step
  "Gibbs sampling step: hidden ---> visible ---> hidden
  returns the expected values and samples of both the visible samples given
  the hidden and the new hidden input and expected values"
  [& {:keys [rbm hidden-input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rbm (_ :guard seq?)
           :hidden-input (:or (_ :guard seq?)
                              (_ :guard vector?))}]
         (obj-or-code? as-code? `(.gibbhVh ~rbm (vec-or-matrix->indarray ~hidden-input)))
         :else
         (let [[h-i rbm-model] (eval-if-code [hidden-input seq?] [rbm seq?])]
           (.gibbhVh rbm-model (vec-or-matrix->indarray h-i)))))

(defn prop-up
  "Calculates the activation of the visible : sigmoid(v * W + hbias)"
  [& {:keys [rbm visible-input training? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rbm (_ :guard seq?)
           :visible-input (:or (_ :guard seq?)
                               (_ :guard vector?))
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.propUp ~rbm (vec-or-matrix->indarray ~visible-input) ~training?))
         [{:rbm _
           :visible-input _
           :training? _}]
         (let [[v-i t? rbm-model] (eval-if-code [visible-input seq?]
                                                [training? seq? boolean?]
                                                [rbm seq?])]
           (.propUp rbm-model (vec-or-matrix->indarray v-i) t?))
         [{:rbm (_ :guard seq?)
           :visible-input (:or (_ :guard seq?)
                               (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          (obj-or-code? as-code? `(.propUp ~rbm (vec-or-matrix->indarray ~visible-input))))
         :else
         (let [[v-i rbm-model] (eval-if-code [visible-input seq?]
                                             [rbm seq?])]
           (.propUp rbm-model (vec-or-matrix->indarray v-i)))))

(defn prop-up-derivative
  "derivative of the prop-up activation"
  [& {:keys [rbm prop-up-vals as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rbm (_ :guard seq?)
           :prop-up-vals (:or (_ :guard seq?)
                              (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(.propUpDerivative ~rbm (vec-or-matrix->indarray ~prop-up-vals)))
         :else
         (let [[p-up rbm-model] (eval-if-code [prop-up-vals seq?]
                                              [rbm seq?])]
           (.propUpDerivative rbm-model (vec-or-matrix->indarray p-up)))))

(defn prop-down
  "Calculates the activation of the hidden: (activation (h * W + vbias))"
  [& {:keys [rbm hidden-input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rbm (_ :guard seq?)
           :hidden-input (:or (_ :guard seq?)
                              (_ :guard vector?))}]
         (obj-or-code? as-code? `(.propDown ~rbm (vec-or-matrix->indarray ~hidden-input)))
         :else
         (let [[h-i rbm-model] (eval-if-code [hidden-input seq?] [rbm seq?])]
           (.propDown rbm-model (vec-or-matrix->indarray h-i)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; frozen layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn log-test-mode
  ":training? (boolean) training or testing?

  :training-mode (keyword) one of :training or :testing"
  [& {:keys [frozen-layer training? training-mode as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:frozen-layer (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~frozen-layer (.logTestMode ~training?)))
         [{:frozen-layer _ :training? _}]
         (let [[t?] (eval-if-code [training? seq? boolean?])]
           (doto frozen-layer (.logTestMode t?)))
         [{:frozen-layer (_ :guard seq?)
           :training-mode (:or (_ :guard keyword?)
                               (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~frozen-layer
            (.logTestMode (enum/value-of {:layer-training-mode ~training-mode}))))
         [{:frozen-layer _ :training-mode _}]
         (let [[t-mode] (eval-if-code [training-mode seq? keyword?])]
          (doto frozen-layer
            (.logTestMode (enum/value-of {:layer-training-mode t-mode}))))))

(defn get-inside-layer
  [frozen-layer & {:keys [as-code?]
                   :or {as-code? true}}]
  (match [frozen-layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getInsideLayer ~frozen-layer))
         :else
         (.getInsideLayer frozen-layer)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; batch normalization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-shape
  "returns an integer array of the shape after passing through the normalization layer"
  [& {:keys [batch-norm features as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:batch-norm (_ :guard seq?)
           :features (:or (_ :guard seq?)
                          (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(.getShape ~batch-norm (vec-or-matrix->indarray ~features)))
         :else
         (let [[f b-norm] (eval-if-code [features seq?] [batch-norm seq?])]
           (.getShape b-norm (vec-or-matrix->indarray f)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; vae
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn is-pretrain-param?
  "checks to see if the supplied param can be pretrained

  :param (string), name of the paramter"
  [& {:keys [vae param as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :param (:or (_ :guard string?)
                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.isPretrainParam ~vae ~param))
         :else
         (let [[p] (eval-if-code [param seq? string?])]
           (.isPretrainParam vae p))))

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
  [& {:keys [vae data num-samples as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :data (:or (_ :guard seq?)
                      (_ :guard vector?))
           :num-samples (:or (_ :guard number?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.reconstructionProbability ~vae (vec-or-matrix->indarray ~data) (int ~num-samples)))
         :else
         (let [[d n-samples model] (eval-if-code [data seq?]
                                                 [num-samples seq? number?]
                                                 [vae seq?])]
           (.reconstructionProbability model (vec-or-matrix->indarray d) n-samples))))

(defn reconstruction-log-probability
  "Return the log reconstruction probability given the specified number of samples.

  the docs in reconstruction-probability also apply to this fn."
  [& {:keys [vae data num-samples as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :data (:or (_ :guard seq?)
                      (_ :guard vector?))
           :num-samples (:or (_ :guard number?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.reconstructionLogProbability ~vae (vec-or-matrix->indarray ~data)
                                         (int ~num-samples)))
         :else
         (let [[d n-samples model] (eval-if-code [data seq?]
                                                 [num-samples seq? number?]
                                                 [vae seq?])]
           (.reconstructionLogProbability model (vec-or-matrix->indarray d) n-samples))))

(defn generate-random-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  randomly generate output x, where x ~ P(x|z)

  :latent-space-values (INDArray or vec), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :latent-space-values (:or (_ :guard seq?)
                                     (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(.generateRandomGivenZ ~vae (vec-or-matrix->indarray ~latent-space-values)))
         :else
         (let [[lsv model] (eval-if-code [latent-space-values seq?]
                                         [vae seq?])]
           (.generateRandomGivenZ model (vec-or-matrix->indarray lsv)))))

(defn generate-at-mean-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  generate output from P(x|z), where x = E[P(x|z)]
   - i.e., return the mean value for the distribution P(x|z)

  :latent-space-values (INDArray or vec), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :latent-space-values (:or (_ :guard seq?)
                                     (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(.generateAtMeanGivenZ ~vae (vec-or-matrix->indarray ~latent-space-values)))
         :else
         (let [[lsv model] (eval-if-code [latent-space-values seq?]
                                         [vae seq?])]
           (.generateAtMeanGivenZ model (vec-or-matrix->indarray lsv)))))

(defn has-loss-fn?
  "Does the reconstruction distribution have a loss function (such as mean squared error)
  or is it a standard probabilistic reconstruction distribution?"
  [vae & {:keys [as-code?]
          :or {as-code? true}}]
  (match [vae]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.hasLossFunction ~vae))
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
  [& {:keys [vae data as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:vae (_ :guard seq?)
           :data (:or (_ :guard seq?)
                      (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(.reconstructionError ~vae (vec-or-matrix->indarray ~data)))
         :else
         (let [[d model] (eval-if-code [data seq?] [vae seq?])]
           (.reconstructionError model (vec-or-matrix->indarray d)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; recurrent layers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn rnn-activate-using-stored-state
  "Similar to rnnTimeStep, this method is used for activations using the state
  stored in the stateMap as the initialization.

  input (INDArray or vec) of input values to the layer
  store-last-for-tbptt? (boolean), save state to be used in tbptt
  training? (boolean) is the model currently in training or not?"
  [& {:keys [rnn-layer input training? store-last-for-tbptt? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :input (:or (_ :guard seq?)
                       (_ :guard vector?))
           :training? (:or (_ :guard seq?)
                           (_ :guard boolean?))
           :store-last-for-tbptt? (:or (_ :guard seq?)
                                       (_ :guard boolean?))}]
         (obj-or-code?
          as-code?
          `(.rnnActivateUsingStoredState rnn-layer (vec-or-matrix->indarray input)
                                        training? store-last-for-tbptt?))
         :else
         (let [[rnn i t? store?] (eval-if-code [rnn-layer seq?] [input seq?]
                                               [training? seq? boolean?]
                                               [store-last-for-tbptt? seq? boolean?])]
          (.rnnActivateUsingStoredState rnn (vec-or-matrix->indarray i) t? store?))))

(defn rnn-clear-previous-state!
  "Reset/clear the stateMap for rnn-time-step and
  tBpttStateMap for rnn-activate-using-stored-state

  returns the rnn-layer"
  [rnn-layer & {:keys [as-code?]
                :or {as-code? true}}]
  (match [rnn-layer]
         [(_ :guard seq?)]
         (obj-or-code?
          as-code?
          `(doto ~rnn-layer
            (.rnnClearPreviousState)))
         :else
         (doto rnn-layer
           (.rnnClearPreviousState))))

(defn rnn-layer-get-prev-state
  "Returns a shallow copy of the RNN stateMap (that contains the stored history
  for use in fns such as rnn-time-step"
  [rnn-layer & {:keys [as-code?]
                :or {as-code? true}}]
  (match [rnn-layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.rnnGetPreviousState ~rnn-layer))
         :else
         (.rnnGetPreviousState rnn-layer)))

(defn rnn-get-tbptt-state
  "Get the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer."
  [rnn-layer & {:keys [as-code?]
                :or {as-code? true}}]
  (match [rnn-layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.rnnGetTBPTTState ~rnn-layer))
         :else
         (.rnnGetTBPTTState rnn-layer)))

(defn rnn-set-tbptt-state!
  "Set the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer.
  and returns the layer

  state is a map of {string (indArray or code for creating one)}"
  [& {:keys [rnn-layer state as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :state (:or (_ :guard map?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~rnn-layer
            (.rnnSetTBPTTState ~state)))
         :else
         (let [[s] (eval-if-code [state seq? map?])]
          (doto rnn-layer (.rnnSetTBPTTState s)))))

(defn rnn-layer-set-prev-state!
  "Set the stateMap (stored history) and return the layer.

  state is a map of {string (indArray or code for creating one)}
   - where seq is the code for creating an indArray

  There is no validation for whats in the state map"
  [& {:keys [rnn-layer state as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :state (:or
                   (_ :guard seq?)
                   (_ :guard map?))}]
         (obj-or-code?
          as-code?
          `(doto ~rnn-layer
            (.rnnSetPreviousState ~state)))
         :else
         (let [[s] (eval-if-code [state seq? map?])]
           (doto rnn-layer (.rnnSetPreviousState s)))))

(defn rnn-layer-time-step
  "Do one or more time steps using the previous time step state stored in stateMap.
  Can be used to efficiently do forward pass one or n-steps at a time (instead of doing forward pass always from t=0)
  If stateMap is empty, default initialization (usually zeros) is used
  Implementations also update stateMap at the end of this method

  input should be an INDArray or vector of input values to the layer"
  [& {:keys [rnn-layer input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :input (:or (_ :guard seq?)
                       (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(.rnnTimeStep ~rnn-layer (vec-or-matrix->indarray ~input)))
         :else
         (let [[rnn i] (eval-if-code [rnn-layer seq?]
                                     [input seq?])]
           (.rnnTimeStep rnn (vec-or-matrix->indarray i)))))

(defn tbptt-backprop-gradient
  "Returns the Truncated BPTT gradient

  epsilon should be an INDArray or vec
  tbptt-back-length is an integer"
  [& {:keys [rnn-layer epsilon tbptt-back-length as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:rnn-layer (_ :guard seq?)
           :epsilon (:or (_ :guard seq?)
                         (_ :guard vector?))
           :tbptt-back-length (:or (_ :guard number?)
                                   (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.tbpttBackpropGradient ~rnn-layer (vec-or-matrix->indarray ~epsilon)
                                  (int ~tbptt-back-length)))
         :else
         (let [[rnn e length] (eval-if-code [rnn-layer seq?]
                                            [epsilon seq?]
                                            [tbptt-back-length seq? number?])]
           (.tbpttBackpropGradient rnn (vec-or-matrix->indarray e) length))))
