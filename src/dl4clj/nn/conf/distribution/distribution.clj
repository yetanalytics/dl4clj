(ns ^{:doc
      "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/Distribution
http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/UniformDistribution.html
http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/NormalDistribution.html
http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/BinomialDistribution.html"}
  dl4clj.nn.conf.distribution.distribution
  (:import [org.deeplearning4j.nn.conf.distribution
            Distribution UniformDistribution NormalDistribution BinomialDistribution
            Distributions GaussianDistribution])
  (:require [dl4clj.utils :refer [generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti distribution generic-dispatching-fn)

(defmethod distribution :uniform [opt]
  (let [config (:uniform opt)
        {l :lower
         u :upper} config]
    (UniformDistribution. l u)))

(defmethod distribution :normal [opt]
  (let [config (:normal opt)
        {m :mean
         std :std} config]
    (NormalDistribution. m std)))

(defmethod distribution :gaussian [opt]
  (let [config (:gaussian opt)
        {m :mean
         std :std} config]
    (GaussianDistribution. m std)))

(defmethod distribution :binomial [opt]
  (let [config (:binomial opt)
        {n-trials :number-of-trials
         prob-success :probability-of-success} config]
    (BinomialDistribution. n-trials prob-success)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-uniform-distribution
  "creates a new uniform distribution with upper and lower bounds

  :upper (double), the upper bound of the distribution

  :lower (double), the lower bound of the distribution"
  [& {:keys [lower upper]
      :as opts}]
  (distribution {:uniform opts}))

(defn new-normal-distribution
  "Create a normal distribution with the given mean and std

  :mean (double), the mean of the distribution

  :std (double), the standard deviation of the distribution"
  [& {:keys [mean std]
      :as opts}]
  (distribution {:normal opts}))

(defn new-gaussian-distribution
  "Create a gaussian distribution with the given mean and std

  :mean (double), the mean of the distribution

  :std (double), the standard deviation of the distribution

  - this is the same thing as creating a normal distribution"
  [& {:keys [mean std]
      :as opts}]
  (distribution {:gaussian opts}))

(defn new-binomial-distribution
  "creates a binomial distribution with the given number of trials and
   probability of success

  :number-of-trials (int), the number of trials

  :probability-of-success (double), how likely an entry within the distribution
   can be classified as a success"
  [& {:keys [number-of-trials probability-of-success]
      :as opts}]
  (distribution {:binomial opts}))
