(ns dl4clj.nn.api.distribution
  (:import [org.deeplearning4j.nn.conf.distribution BinomialDistribution
            NormalDistribution UniformDistribution GaussianDistribution])
  (:require [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; binomial distribution fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-n-trials
  "returns the number of trials set for this distribution"
  [binomial-dist]
  (match [binomial-dist]
         [(_ :guard seq?)]
         `(.getNumberOfTrials ~binomial-dist)
         :else
         (.getNumberOfTrials binomial-dist)))

(defn get-prob-of-success
  "returns the probability of success set for this distribution"
  [binomial-dist]
  (match [binomial-dist]
         [(_ :guard seq?)]
         `(.getProbabilityOfSuccess ~binomial-dist)
         :else
         (.getProbabilityOfSuccess binomial-dist)))

(defn set-prob-of-success!
  "sets the probability of sucess for the provided distribution.

  returns the distribution after the change"
  [& {:keys [binomial-dist prob]
      :as opts}]
  (match [opts]
         [{:binomial-dist (_ :guard seq?)
           :prob (:or (_ :guard seq?)
                      (_ :guard number?))}]
         `(doto ~binomial-dist (.setProbabilityOfSuccess (double ~prob)))
         :else
         (doto binomial-dist (.setProbabilityOfSuccess prob))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; normal/gaussian distribution fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-mean
  "return the mean for the normal/gaussian distribution"
  [dist]
  (match [dist]
         [(_ :guard seq?)]
         `(.getMean ~dist)
         :else
         (.getMean dist)))

(defn get-std
  "return the standard deviation for the normal/gaussian distribution"
  [dist]
  (match [dist]
         [(_ :guard seq?)]
         `(.getStd ~dist)
         :else
         (.getStd dist)))

(defn set-mean!
  "sets the mean for the normal/gaussian distribution supplied.

  :mean (double), the desired mean for the distribution

  returns the distribution after the change"
  [& {:keys [dist mean]
      :as opts}]
  (match [opts]
         [{:dist (_ :guard seq?)
           :mean (:or (_ :guard seq?)
                      (_ :guard number?))}]
         `(doto ~dist (.setMean (double ~mean)))
         :else
         (doto dist (.setMean mean))))

(defn set-std
  "sets the standard deviation for the normal/gaussian distribution supplied.

  :std (double), the desired standard deviation for the distribution

  returns the distribution after the change"
  [& {:keys [dist std]
      :as opts}]
  (match [opts]
         [{:dist (_ :guard seq?)
           :std (:or (_ :guard seq?)
                     (_ :guard number?))}]
         `(doto ~dist (.setStd (double ~std)))
         :else
         (doto dist (.setStd std))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; uniform distribution fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-lower
  "returns the lower bound of the supplied uniform distribution"
  [uniform-dist]
  (match [uniform-dist]
         [(_ :guard seq?)]
         `(.getLower ~uniform-dist)
         :else
         (.getLower uniform-dist)))

(defn get-upper
  "returns the upper bound of the supplied uniform distribution"
  [uniform-dist]
  (match [uniform-dist]
         [(_ :guard seq?)]
         `(.getUpper ~uniform-dist)
         :else
         (.getUpper uniform-dist)))

(defn set-lower!
  "sets the lower bound of the supplied uniform distribution

  :lower (double), the lower bound of the distribution

  returns the distribution after it has changed"
  [& {:keys [uniform-dist lower]
      :as opts}]
  (match [opts]
         [{:uniform-dist (_ :guard seq?)
           :lower (:or (_ :guard seq?)
                       (_ :guard number?))}]
         `(doto ~uniform-dist (.setLower (double ~lower)))
         :else
         (doto uniform-dist (.setLower lower))))

(defn set-upper!
  "sets the upper bound of the supplied uniform distribution

  :upper (double), the upper bound of the distribution

  returns the distribution after it has changed"
  [& {:keys [uniform-dist upper]
      :as opts}]
  (match [opts]
         [{:uniform-dist (_ :guard seq?)
           :upper (:or (_ :guard seq?)
                       (_ :guard number?))}]
         `(doto ~uniform-dist (.setUpper (double ~upper)))
         :else
         (doto uniform-dist (.setUpper upper))))
