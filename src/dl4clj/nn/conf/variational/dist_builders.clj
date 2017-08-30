(ns dl4clj.nn.conf.variational.dist-builders
  (:import [org.deeplearning4j.nn.conf.layers.variational
            BernoulliReconstructionDistribution
            CompositeReconstructionDistribution
            CompositeReconstructionDistribution$Builder
            ExponentialReconstructionDistribution
            GaussianReconstructionDistribution])
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many? builder-fn
                                  eval-and-build]]
            [dl4clj.constants :as enum]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method for creating the distribution objects
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; update to have mm produce code, user facing determines if obj is created
(defmulti distributions generic-dispatching-fn)

(defmethod distributions :exponential [opts]
  (let [conf (:exponential opts)
        activation-fn (:activation-fn conf)]
    (if (contains? conf :activation-fn)
      (ExponentialReconstructionDistribution.
       (enum/value-of {:activation-fn activation-fn}))
      (ExponentialReconstructionDistribution.))))

(defmethod distributions :bernoulli [opts]
  (let [conf (:bernoulli opts)
        activation-fn (:activation-fn conf)]
    (if (contains? conf :activation-fn)
      (BernoulliReconstructionDistribution.
       (enum/value-of {:activation-fn activation-fn}))
      (BernoulliReconstructionDistribution.))))

(defmethod distributions :gaussian [opts]
  (let [conf (:gaussian opts)
        activation-fn (:activation-fn conf)]
    (if (contains? conf :activation-fn)
      (GaussianReconstructionDistribution.
       (enum/value-of {:activation-fn activation-fn}))
      (GaussianReconstructionDistribution.))))

(defmethod distributions :composite [opts]
  (let [conf (:composite opts)
        dists (:distributions-to-add conf)
        m-calls (into []
                      (for [each dists
                            :let [k (first (keys each))
                                  data (k each)
                                  dist-size (:dist-size data)]]
                        [dist-size `(distributions {~k ~data})]))]
    (eval-and-build (builder-fn `(CompositeReconstructionDistribution$Builder.)
                                {:to-add '.addDistribution}
                                {:to-add m-calls}))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns which provide documentation on distribution creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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
  [distributions-to-add]
  `(distributions {:composite {:distributions-to-add ~distributions-to-add}}))

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
  [& {:keys [activation-fn]
      :as opts}]
  `(distributions {:bernoulli ~opts}))

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
  [& {:keys [activation-fn]
      :as opts}]
  `(distributions {:exponential ~opts}))

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
  [& {:keys [activation-fn]
      :as opts}]
  `(distributions {:gaussian ~opts}))
