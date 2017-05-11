(ns ^{:doc "Variational Autoencoder layer
See: Kingma & Welling, 2013: Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114

This implementation allows multiple encoder and decoder layers, the number and sizes of which can be set independently.

see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/variational/VariationalAutoencoder.html"}
    dl4clj.nn.layers.variational-autoencoder
  (:import [org.deeplearning4j.nn.layers.variational VariationalAutoencoder]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; constructor
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-variational-autoencoder
  [& {:keys [conf]}]
  (VariationalAutoencoder. conf))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; interaction fns unique to VAEs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn generate-at-mean-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  generate output from P(x|z), where x = E[P(x|z)]
   - i.e., return the mean value for the distribution P(x|z)

  :latent-space-values (INDArray), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values]}]
  (.generateAtMeanGivenZ vae latent-space-values))

(defn generate-random-given-z
  "Given a specified values for the latent space as input (latent space being z in p(z|data)),
  randomly generate output x, where x ~ P(x|z)

  :latent-space-values (INDArray), Values for the latent space.
    - size(1) must equal nOut configuration parameter

  :vae (layer), is the variational autoencoder"
  [& {:keys [vae latent-space-values]}]
  (.generateRandomGivenZ vae latent-space-values))

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

(defn is-pretrain-param?
  "checks to see if the supplied param can be pretrained

  :param (string), name of the paramter"
  [& {:keys [vae param]}]
  (.isPretrainParam vae param))

(defn pre-output
  "get the pre training output of the supplied vae"
  [& {:keys [vae training?]}]
  (.preOutput vae training?))
