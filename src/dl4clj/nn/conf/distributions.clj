(ns ^{:doc
      "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/Distribution
http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/UniformDistribution.html
http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/NormalDistribution.html
http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/BinomialDistribution.html"}
  dl4clj.nn.conf.distributions
  (:import [org.deeplearning4j.nn.conf.distribution
            Distribution UniformDistribution NormalDistribution BinomialDistribution
            Distributions GaussianDistribution]
           [org.deeplearning4j.nn.conf.layers.variational
            BernoulliReconstructionDistribution
            CompositeReconstructionDistribution
            CompositeReconstructionDistribution$Builder
            ExponentialReconstructionDistribution
            GaussianReconstructionDistribution])
  (:require [dl4clj.utils :refer [generic-dispatching-fn obj-or-code? builder-fn]]
            [dl4clj.constants :as enum]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; update to have mm produce code, user facing determines if obj is created
(defmulti distribution generic-dispatching-fn)

(defmethod distribution :uniform [opt]
  (let [config (:uniform opt)
        {l :lower
         u :upper} config]
    `(UniformDistribution. ~l ~u)))

(defmethod distribution :normal [opt]
  (let [config (:normal opt)
        {m :mean
         std :std} config]
    `(NormalDistribution. ~m ~std)))

(defmethod distribution :gaussian [opt]
  (let [config (:gaussian opt)
        {m :mean
         std :std} config]
    `(GaussianDistribution. ~m ~std)))

(defmethod distribution :binomial [opt]
  (let [config (:binomial opt)
        {n-trials :number-of-trials
         prob-success :probability-of-success} config]
    `(BinomialDistribution. ~n-trials ~prob-success)))

(defmethod distribution :exponential [opts]
  (let [conf (:exponential opts)
        activation-fn (:activation-fn conf)]
    (if (contains? conf :activation-fn)
      `(ExponentialReconstructionDistribution.
        (enum/value-of {:activation-fn ~activation-fn}))
      `(ExponentialReconstructionDistribution.))))

(defmethod distribution :bernoulli [opts]
  (let [conf (:bernoulli opts)
        activation-fn (:activation-fn conf)]
    (if (contains? conf :activation-fn)
      `(BernoulliReconstructionDistribution.
        (enum/value-of {:activation-fn ~activation-fn}))
      `(BernoulliReconstructionDistribution.))))

(defmethod distribution :gaussian-reconstruction [opts]
  (let [conf (:gaussian-reconstruction opts)
        activation-fn (:activation-fn conf)]
    (if (contains? conf :activation-fn)
      `(GaussianReconstructionDistribution.
        (enum/value-of {:activation-fn ~activation-fn}))
      `(GaussianReconstructionDistribution.))))

(defmethod distribution :composite [opts]
  (let [conf (:composite opts)
        dists (:distributions conf)
        m-calls (into []
                      (for [each dists
                            :let [k (first (keys each))
                                  data (k each)
                                  dist-size (:dist-size data)]]
                        [dist-size (distribution {k data})]))]
    `(.build ~(builder-fn `(CompositeReconstructionDistribution$Builder.)
                          {:to-add '.addDistribution}
                          {:to-add m-calls}))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns (may be removed with the change to how nns are built)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-uniform-distribution
  "creates a new uniform distribution with upper and lower bounds

  :upper (double), the upper bound of the distribution

  :lower (double), the lower bound of the distribution"
  [& {:keys [lower upper as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (distribution {:uniform opts})]
    (obj-or-code? as-code? code)))

(defn new-normal-distribution
  "Create a normal distribution with the given mean and std

  :mean (double), the mean of the distribution

  :std (double), the standard deviation of the distribution"
  [& {:keys [mean std as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (distribution {:normal opts})]
    (obj-or-code? as-code? code)))

(defn new-gaussian-distribution
  "Create a gaussian distribution with the given mean and std

  :mean (double), the mean of the distribution

  :std (double), the standard deviation of the distribution

  - this is the same thing as creating a normal distribution"
  [& {:keys [mean std as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (distribution {:gaussian opts})]
    (obj-or-code? as-code? code)))

(defn new-binomial-distribution
  "creates a binomial distribution with the given number of trials and
   probability of success

  :number-of-trials (int), the number of trials

  :probability-of-success (double), how likely an entry within the distribution
   can be classified as a success"
  [& {:keys [number-of-trials probability-of-success as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (distribution {:binomial opts})]
    (obj-or-code? as-code? code)))

(defn new-composite-reconstruction-distribution
  "creates a new composite reconstruction distributions
   - a combination of multiple reconstruction distributions

  distributions-to-add (vector) [{dist1 config} {dist2 config}]
   - distribution take the form of:
      {dist-type (keyword) {:activation-fn keyword :dist-size int}}
      - see other fns for avialable activation fns (optional config)

  - distribution params pairs (:dist-size needs to be specifed in both cases)
     - :dist-tyye (keyword) the type of distribution you want to add to your composition
        - possible keywords are: :bernoulli, :exponential, :gaussian

     - :dist-opts (map), the map of options required for constructing a distribution
        - see the desired distribution new fn for required options

     - :dist-size (int), number of values to be modelled by the supplied distribution (inclusive)

  ex. args: [{:bernoulli {:activation-fn :tanh :dist-size 5}}
             {:bernoulli {:dist-size 10}}]
   would result in the code to create a composite distribution containing both distributions specified above

  the result will need to be evaluated and built at the proper time"
  [& {:keys [distributions as-code?]
      :or {as-code? true}}]
  (let [code `(distribution {:composite {:distributions ~distributions}})]
    (obj-or-code? as-code? code)))

(defn new-bernoulli-reconstruction-distribution
  "Bernoulli reconstruction distribution for variational autoencoder.
  Outputs are modelled by a Bernoulli distribution
  - i.e., the Bernoulli distribution should be used for binary data
  (all values 0 or 1)
  the VAE models the probability of the output being 0 or 1.

  Consequently, the sigmoid activation function should be used to bound
  activations to the range of 0 to 1. Activation functions that do not
  produce outputs in the range of 0 to 1 (including relu, tanh, and many others)
  should be avoided.

  :activation-fn (keyword), the activation fn to be used with the dist
   -see https://deeplearning4j.org/features#activation-functions"
  [& {:keys [activation-fn as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (distribution {:bernoulli opts})]
    (obj-or-code? as-code? code)))

(defn new-exponential-reconstruction-distribution
  "Exponential reconstruction distribution. Supports data in range 0-infinity

  Parameterization used here:
  - network models distribution parameter gamma,
    - gamma = log(lambda), with gamma in (-inf, inf)

  This means that an input from the decoder of gamma = 0 gives lambda = 1
   - corresponds to a mean value for the expontial distribution of 1/lambda = 1

  Regarding the choice of activation function:
  - the parameterization above supports gamma in the range (-infinity,infinity)
    therefore a symmetric activation function such as identity or perhaps tanh is preferred.

  :activation-fn (keyword), the activation fn to be used with the dist
   -see https://deeplearning4j.org/features#activation-functions"
  [& {:keys [activation-fn as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (distribution {:exponential opts})]
    (obj-or-code? as-code? code)))

(defn new-gaussian-reconstruction-distribution
  "Gaussian reconstruction distribution for variational autoencoder.

  Outputs are modelled by a Gaussian distribution, with the mean and variances
  (diagonal covariance matrix) for each output determined by the network forward pass.

   - Specifically, the GaussianReconstructionDistribution models mean and log(stdev^2).
     This parameterization gives log(1) = 0, and inputs can be in range (-infinity,infinity).

   - Other parameterizations for variance are of course possible but may be problematic
     with respect to the average pre-activation function values and activation function ranges.

   - For activation functions, identity and perhaps tanh are typical
      -though tanh (unlike identity) implies a minimum/maximum possible value for mean and log variance.

   - Asymmetric activation functions such as sigmoid or relu should be avoided.

  :activation-fn (keyword), the activation fn to be used with the dist
   -see https://deeplearning4j.org/features#activation-functions"
  [& {:keys [activation-fn as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (distribution {:gaussian-reconstruction opts})]
    (obj-or-code? as-code? code)))
