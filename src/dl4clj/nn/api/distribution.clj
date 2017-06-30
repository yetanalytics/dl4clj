(ns dl4clj.nn.api.distribution
  (:import [org.deeplearning4j.nn.conf.distribution BinomialDistribution
            NormalDistribution UniformDistribution GaussianDistribution]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; binomial distribution fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-n-trials
  "returns the number of trials set for this distribution"
  [binomial-dist]
  (.getNumberOfTrials binomial-dist))

(defn get-prob-of-success
  "returns the probability of success set for this distribution"
  [binomial-dist]
  (.getProbabilityOfSuccess binomial-dist))

(defn set-prob-of-success!
  "sets the probability of sucess for the provided distribution.

  returns the distribution after the change"
  [& {:keys [binomial-dist prob]}]
  (doto binomial-dist (.setProbabilityOfSuccess prob)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; normal/gaussian distribution fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-mean
  "return the mean for the normal/gaussian distribution"
  [dist]
  (.getMean dist))

(defn get-std
  "return the standard deviation for the normal/gaussian distribution"
  [dist]
  (.getStd dist))

(defn set-mean!
  "sets the mean for the normal/gaussian distribution supplied.

  :mean (double), the desired mean for the distribution

  returns the distribution after the change"
  [& {:keys [dist mean]}]
  (doto dist (.setMean mean)))

(defn set-std
  "sets the standard deviation for the normal/gaussian distribution supplied.

  :std (double), the desired standard deviation for the distribution

  returns the distribution after the change"
  [& {:keys [dist std]}]
  (doto dist (.setStd std)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; uniform distribution fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-lower
  "returns the lower bound of the supplied uniform distribution"
  [uniform-dist]
  (.getLower uniform-dist))

(defn get-upper
  "returns the upper bound of the supplied uniform distribution"
  [uniform-dist]
  (.getUpper uniform-dist))

(defn set-lower!
  "sets the lower bound of the supplied uniform distribution

  :lower (double), the lower bound of the distribution

  returns the distribution after it has changed"
  [& {:keys [uniform-dist lower]}]
  (doto uniform-dist (.setLower lower)))

(defn set-upper!
  "sets the upper bound of the supplied uniform distribution

  :upper (double), the upper bound of the distribution

  returns the distribution after it has changed"
  [& {:keys [uniform-dist upper]}]
  (doto uniform-dist (.setUpper upper)))
