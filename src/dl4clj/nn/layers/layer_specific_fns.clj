(ns dl4clj.nn.layers.layer-specific-fns
  (:import [org.deeplearning4j.nn.layers BaseOutputLayer
            BasePretrainNetwork FrozenLayer LossLayer]
           [org.deeplearning4j.nn.layers.feedforward.autoencoder AutoEncoder]
           [org.deeplearning4j.nn.layers.normalization BatchNormalization]
           [org.deeplearning4j.nn.layers.variational VariationalAutoencoder]
           [org.deeplearning4j.nn.layers.feedforward.rbm RBM])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.nn.conf.constants :as enum]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; base output layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn layer-output
  "returns the output of a layer.

  :input (INDArray), the input to the layer

  :training? (boolean), are we in trianing mode or testing?

  multiple layers implement this fn"
  [& {:keys [layer input training?]
      :as opts}]
  (cond (contains-many? opts :input :training?)
        (.output layer input training?)
        (contains? opts :input)
        (.output layer input)
        (contains? opts :training?)
        (.output layer training?)
        :else
        (assert false "you must supply a layer and either some input or training?")))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; base pretrain layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-corrupted-input
  "Corrupts the given input by doing a binomial sampling given the corruption level

  :features is an INDArray of input features to the network

  :corruption-level (double) amount of corruption to apply."
  [& {:keys [base-pretrain-network features corruption-level]}]
  (.getCorruptedInput base-pretrain-network features corruption-level))

(defn sample-hidden-given-visible
  "Sample the hidden distribution given the visible

  :visible is an INDArray of the visibile distribution"
  [& {:keys [base-pretrain-network visible]}]
  (.sampleHiddenGivenVisible base-pretrain-network visible))

(defn sample-visible-given-hidden
  "Sample the visible distribution given the hidden

  :hidden is an INDArray of the hidden distribution"
  [& {:keys [base-pretrain-network hidden]}]
  (.sampleVisibleGivenHidden base-pretrain-network hidden))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; autoencoders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn decode
  "decodes an encoded output of an autoencoder"
  [& {:keys [autoencoder layer-output]}]
  (.decode autoencoder layer-output))

(defn encode
  "encodes an input to be passed to an autoencoder"
  [& {:keys [autoencoder input training?]}]
  (.encode autoencoder input training?))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rbm
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn gibbs-sampling-step
  "Gibbs sampling step: hidden ---> visible ---> hidden
  returns the expected values and samples of both the visible samples given
  the hidden and the new hidden input and expected values"
  [& {:keys [rbm hidden-input]}]
  (.gibbhVh rbm hidden-input))

(defn prop-up
  "Calculates the activation of the visible : sigmoid(v * W + hbias)"
  [& {:keys [rbm visible-input training?]
      :as opts}]
  (assert (contains-many? opts :rbm :visible-input)
          "you must supply a rbm and the visible input")
  (if (contains? opts :training?)
    (.propUp rbm visible-input training?)
    (.propUp rbm visible-input)))

(defn prop-up-derivative
  "derivative of the prop-up activation"
  [& {:keys [rbm prop-up-vals]}]
  (.propUpDerivative rbm prop-up-vals))

(defn prop-down
  "Calculates the activation of the hidden: (activation (h * W + vbias))"
  [& {:keys [rbm hidden-input]}]
  (.propDown rbm hidden-input))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; frozen layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn log-test-mode
  ;; figure out exactly what this does. no docs for this method
  ":training? (boolean) training or testing?

  :training-mode (keyword) one of :training or :testing"
  [& {:keys [frozen-layer training? training-mode]
      :as opts}]
  (assert (contains? opts :frozen-layer) "you must supply a frozen layer")
  (cond (contains? opts :training?)
        (doto frozen-layer
          (.logTestMode training?))
        (contains? opts :training-mode)
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
  (.getShape batch-norm features))

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

  :data (INDArray), the data to calculate the reconstruction probability for

  :num-samples (int), Number of samples with which to base the reconstruction probability on."
  [& {:keys [vae data num-samples]}]
  (.reconstructionProbability vae data num-samples))

(defn reconstruction-log-probability
  "Return the log reconstruction probability given the specified number of samples.

  the docs in reconstruction-probability also apply to this fn."
  [& {:keys [vae data num-samples]}]
  (.reconstructionLogProbability vae data num-samples))

(defn generate-random-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  randomly generate output x, where x ~ P(x|z)

  :latent-space-values (INDArray), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values]}]
  (.generateRandomGivenZ vae latent-space-values))

(defn generate-at-mean-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  generate output from P(x|z), where x = E[P(x|z)]
   - i.e., return the mean value for the distribution P(x|z)

  :latent-space-values (INDArray), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values]}]
  (.generateAtMeanGivenZ vae latent-space-values))

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

  :data (INDArray), the data to calc the reconstruction error on"
  [& {:keys [vae data]}]
  (.reconstructionError vae data))
