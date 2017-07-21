(ns dl4clj.nn.conf.variational.dist-builders
  (:import [org.deeplearning4j.nn.conf.layers.variational
            BernoulliReconstructionDistribution
            CompositeReconstructionDistribution
            CompositeReconstructionDistribution$Builder
            ExponentialReconstructionDistribution
            GaussianReconstructionDistribution])
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many? builder-fn
                                  builder-fn-repeated-method-call]]
            [dl4clj.constants :as enum]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method for creating the distribution objects
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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
        data-in-vec (into [] (for [each dists]
                               each))]
    (loop [builder (CompositeReconstructionDistribution$Builder.)
           distz! data-in-vec]
      (let [the-dist (first distz!)
            ;; get the data
            [dist-idx dist-data] the-dist
            dist-conf (first dist-data)
            [dist-type dist-opts] dist-conf
            dist-size (:dist-size dist-opts)]
        (cond (empty? distz!)
              ;; we have gone through all our distributions
              (.build builder)
              :else
              (recur
               ;; we create the distribution and add it to the builder
               (.addDistribution builder dist-size (distributions {dist-type dist-opts}))
               (rest distz!)))))))

;; come back and finish this

#_(defn test-composite
  [opts]
 (let [conf (:composite opts)]
   (for [each conf
         :let [
               dist-type (keys each)
               dist-opts (vals each)
               [data] dist-opts
               size (:dist-size data)
               a-fn (:activation-fn data)
               #_[dist-fn dist-size] #_dist-opts]]
     a-fn)))

#_(test-composite
 {:composite
  {0 {:bernoulli {:activation-fn :sigmoid
               :dist-size     5}}
   1 {:exponential {:activation-fn :sigmoid
                 :dist-size     3}}
   2 {:gaussian {:activation-fn :hard-tanh
              :dist-size     1}}
   3 {:bernoulli {:activation-fn :sigmoid
               :dist-size     4}}
   4 {:bernoulli {:dist-size     4}}}})

#_(builder-fn-repeated-method-call
 `(CompositeReconstructionDistribution$Builder.)
 {:add '.addDistribution}
 {:add {5 '(distributions {:bernoulli {:activation-fn :sigmoid}})
        7 '(distributions {:bernoulli {:activation-fn :sigmoid}})}})

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns which provide documentation on distribution creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-composite-reconstruction-distribution
  "creates a new composite reconstruction distributions
   - a combination of multiple reconstruction distributions

  distributions-to-add (map) {unique-place-holder-1 distribution
                               unique-place-holder-2 distribution ...}
   - distribution can take the form of:
      {dist-type (keyword) {dist-opts :dist-size int}}
      - when you want to create a new distribution with supplied dist-type/opts
     or
      {:dist built-distribution :dist-size int}
      - when you want to add a preexisting distribution to the composite

  - distribution params pairs (:dist-size needs to be specifed in both cases)
     - :dist-tyye (keyword) the type of distribution you want to add to your composition
        - possible keywords are: :bernoulli, :exponential, :gaussian

     - :dist-opts (map), the map of options required for constructing a distribution
        - see the desired distribution new fn for required options

     - :dist (object), the dl4j distribution object

     - :dist-size (int), number of values to be modelled by the supplied distribution (inclusive)

  - unique-place-holder (any-type), could be an int, string, double, long ...,
      just needs to be unique to the distribution that follows
       - this value is not used in computation but is needed for uniqueness per dist
       - recomend just using incrementing integers (0,1,2,3,...)

  ex. args: :distributions-to-add {0 {:dist some-existing-dist-dl4j-object
                                      :dist-size size-of-existing-dist-object}
                                   1 {:bernoulli {:activation-fn :tanh
                                                  :dist-size 5}}}
   would result in a composite distribution containing both distributions specified above

  if still confused, see the example at the bottom of this namespace"
  [distributions-to-add]
  (distributions {:composite {:distributions-to-add distributions-to-add}}))

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
  (distributions {:bernoulli opts}))

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
  (distributions {:exponential opts}))

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
  (distributions {:gaussian opts}))
