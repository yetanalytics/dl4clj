(ns dl4clj.nn.api.distribution
  (:import [org.deeplearning4j.nn.conf.distribution BinomialDistribution
            NormalDistribution UniformDistribution GaussianDistribution])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; binomial distribution fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-n-trials
  "returns the number of trials set for this distribution"
  [binomial-dist & {:keys [as-code?]
                    :or {as-code? true}}]
  (match [binomial-dist]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getNumberOfTrials ~binomial-dist))
         :else
         (.getNumberOfTrials binomial-dist)))

(defn get-prob-of-success
  "returns the probability of success set for this distribution"
  [binomial-dist & {:keys [as-code?]
                    :or {as-code? true}}]
  (match [binomial-dist]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getProbabilityOfSuccess ~binomial-dist))
         :else
         (.getProbabilityOfSuccess binomial-dist)))

(defn set-prob-of-success!
  "sets the probability of sucess for the provided distribution.

  returns the distribution after the change"
  [& {:keys [binomial-dist prob as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:binomial-dist (_ :guard seq?)
           :prob (:or (_ :guard seq?)
                      (_ :guard number?))}]
         (obj-or-code?
          as-code?
          `(doto ~binomial-dist (.setProbabilityOfSuccess (double ~prob))))
         :else
         (doto binomial-dist (.setProbabilityOfSuccess prob))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; normal/gaussian distribution fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-mean
  "return the mean for the normal/gaussian distribution"
  [normal-dist & {:keys [as-code?]
                  :or {as-code? true}}]
  (match [normal-dist]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getMean ~normal-dist))
         :else
         (.getMean normal-dist)))

(defn get-std
  "return the standard deviation for the normal/gaussian distribution"
  [normal-dist & {:keys [as-code?]
                  :or {as-code? true}}]
  (match [normal-dist]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getStd ~normal-dist))
         :else
         (.getStd normal-dist)))

(defn set-mean!
  "sets the mean for the normal/gaussian distribution supplied.

  :mean (double), the desired mean for the distribution

  returns the distribution after the change"
  [& {:keys [normal-dist mean as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:normal-dist (_ :guard seq?)
           :mean (:or (_ :guard seq?)
                      (_ :guard number?))}]
         (obj-or-code? as-code? `(doto ~normal-dist (.setMean (double ~mean))))
         :else
         (doto normal-dist (.setMean mean))))

(defn set-std
  "sets the standard deviation for the normal/gaussian distribution supplied.

  :std (double), the desired standard deviation for the distribution

  returns the distribution after the change"
  [& {:keys [normal-dist std as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:normal-dist (_ :guard seq?)
           :std (:or (_ :guard seq?)
                     (_ :guard number?))}]
         (obj-or-code? as-code? `(doto ~normal-dist (.setStd (double ~std))))
         :else
         (doto normal-dist (.setStd std))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; uniform distribution fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-lower
  "returns the lower bound of the supplied uniform distribution"
  [uniform-dist & {:keys [as-code?]
                   :or {as-code? true}}]
  (match [uniform-dist]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getLower ~uniform-dist))
         :else
         (.getLower uniform-dist)))

(defn get-upper
  "returns the upper bound of the supplied uniform distribution"
  [uniform-dist & {:keys [as-code?]
                   :or {as-code? true}}]
  (match [uniform-dist]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getUpper ~uniform-dist))
         :else
         (.getUpper uniform-dist)))

(defn set-lower!
  "sets the lower bound of the supplied uniform distribution

  :lower (double), the lower bound of the distribution

  returns the distribution after it has changed"
  [& {:keys [uniform-dist lower as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:uniform-dist (_ :guard seq?)
           :lower (:or (_ :guard seq?)
                       (_ :guard number?))}]
         (obj-or-code? as-code? `(doto ~uniform-dist (.setLower (double ~lower))))
         :else
         (doto uniform-dist (.setLower lower))))

(defn set-upper!
  "sets the upper bound of the supplied uniform distribution

  :upper (double), the upper bound of the distribution

  returns the distribution after it has changed"
  [& {:keys [uniform-dist upper as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:uniform-dist (_ :guard seq?)
           :upper (:or (_ :guard seq?)
                       (_ :guard number?))}]
         (obj-or-code? as-code? `(doto ~uniform-dist (.setUpper (double ~upper))))
         :else
         (doto uniform-dist (.setUpper upper))))
